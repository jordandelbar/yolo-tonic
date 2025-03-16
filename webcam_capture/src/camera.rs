use crate::bounding_box::BoundingBoxWithLabels;
use crate::config::CameraPollingConfig;
use crate::cv_utils::{CvUtilsError, ImageConverter};
use crate::prediction::{PredictionService, PredictionServiceError};
use opencv::{core::Mat, prelude::*, videoio};
use std::{sync::Arc, time::Duration};
use thiserror::Error;
use tokio::{
    sync::{broadcast, Mutex},
    time::sleep,
};

#[derive(Error, Debug)]
pub enum CameraError {
    #[error("Failed to open camera: {0}")]
    OpenCameraFailed(opencv::Error),
    #[error("Failed to read frame: {0}")]
    ReadFrameFailed(opencv::Error),
    #[error("OpenCV error: {0}")]
    OpenCvError(opencv::Error),
    #[error("Prediction service error: {0}")]
    PredictionError(#[from] PredictionServiceError),
    #[error("Cv utils error: {0}")]
    OpenCvUtilsError(#[from] CvUtilsError),
}

impl From<opencv::Error> for CameraError {
    fn from(err: opencv::Error) -> Self {
        CameraError::OpenCvError(err)
    }
}

#[derive(Debug)]
pub struct Camera {
    pub capture: Mutex<videoio::VideoCapture>,
    pub predictions: Mutex<Vec<BoundingBoxWithLabels>>,
}

impl Camera {
    pub async fn new() -> Result<Self, CameraError> {
        let capture = videoio::VideoCapture::new(0, videoio::CAP_ANY)
            .map_err(CameraError::OpenCameraFailed)?;

        Ok(Self {
            capture: Mutex::new(capture),
            predictions: Mutex::new(vec![]),
        })
    }

    pub async fn capture_frame(&self) -> Result<Mat, CameraError> {
        let mut cam = self.capture.lock().await;
        let mut frame = Mat::default();
        cam.read(&mut frame).map_err(CameraError::ReadFrameFailed)?;
        Ok(frame)
    }

    pub async fn get_annotated_frame(&self) -> Result<Option<Vec<u8>>, CameraError> {
        let mut cam = self.capture.lock().await;
        let mut frame = Mat::default();

        if cam.read(&mut frame).map_err(CameraError::ReadFrameFailed)? && !frame.empty() {
            let predictions = self.predictions.lock().await;

            ImageConverter::annotate_frame(&mut frame, &predictions)?;

            let image_data = ImageConverter::encode_mat_to_jpg(&frame)?;

            return Ok(Some(image_data));
        }
        Ok(None)
    }
}

pub struct CameraPoller {
    camera: Arc<Camera>,
    prediction_service: Arc<PredictionService>,
    poll_interval_ms: u64,
    max_retries: u64,
    initial_delay: u64,
    backoff_factor: u32,
    max_consecutive_failures: u64,
}

impl CameraPoller {
    pub fn new(
        camera: Arc<Camera>,
        prediction_service: Arc<PredictionService>,
        camera_polling_config: &CameraPollingConfig,
    ) -> Self {
        Self {
            camera,
            prediction_service,
            poll_interval_ms: camera_polling_config.get_prediction_delay_ms(),
            max_retries: camera_polling_config.max_retries,
            initial_delay: camera_polling_config.initial_delay,
            backoff_factor: camera_polling_config.backoff_factor,
            max_consecutive_failures: camera_polling_config.max_consecutive_failures,
        }
    }

    pub async fn run(&self, mut shutdown_rx: broadcast::Receiver<()>) {
        let camera = self.camera.clone();
        let prediction_service = self.prediction_service.clone();
        let poll_interval_ms = self.poll_interval_ms;
        let max_retries = self.max_retries;
        let initial_delay = Duration::from_millis(self.initial_delay);
        let backoff_factor = self.backoff_factor;
        let mut consecutive_failures = 0;
        let max_consecutive_failures = self.max_consecutive_failures;

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    result = Self::poll_and_predict(&camera, prediction_service.clone()) => {
                        match result {
                            Ok(_) => {
                                consecutive_failures = 0;
                            },
                            Err(ref err) => {
                                tracing::error!("Error during polling: {:?}", err);
                                let mut retry_delay = initial_delay;
                                let mut retry_successful = false;
                                for retry_count in 0..max_retries {
                                    tracing::warn!(
                                        "Retrying poll (attempt {}/{}) on consecutive failures: {}/{}",
                                        retry_count + 1,
                                        max_retries,
                                        consecutive_failures,
                                        max_consecutive_failures
                                    );
                                    sleep(retry_delay).await;

                                    let retry_result = Self::poll_and_predict(&camera, prediction_service.clone()).await;
                                    if retry_result.is_ok() {
                                        tracing::info!("Retry successful");
                                        retry_successful = true;
                                        break;
                                    } else {
                                        tracing::error!("Retry failed: {:?}", retry_result);
                                        retry_delay *= backoff_factor;
                                    }
                                }
                                if !retry_successful {
                                    consecutive_failures += 1;
                                    tracing::error!("Max number of retries reached, skipping current poll interval");
                                }
                                if consecutive_failures >= max_consecutive_failures {
                                    tracing::error!("Persistent failure detected. Exiting polling loop");
                                    break;
                                }
                            }
                        }
                    },
                    _ = shutdown_rx.recv() => {
                        tracing::info!("Camera polling received shutdown signal");
                        break;
                    }
                }

                sleep(Duration::from_millis(poll_interval_ms)).await;
            }
            tracing::info!("Camera polling stopped");
        });
    }

    async fn poll_and_predict(
        camera: &Arc<Camera>,
        prediction_service: Arc<PredictionService>,
    ) -> Result<(), CameraError> {
        let frame = camera.capture_frame().await?;
        if frame.empty() {
            return Ok(());
        }

        let image_data = ImageConverter::encode_mat_to_jpg(&frame)?;

        let detections = prediction_service.predict(image_data).await?;
        let mut pred_lock = camera.predictions.lock().await;
        *pred_lock = detections;

        Ok(())
    }
}
