use crate::bounding_box::BoundingBoxWithLabels;
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
}

impl CameraPoller {
    pub fn new(
        camera: Arc<Camera>,
        prediction_service: Arc<PredictionService>,
        poll_interval_ms: u64,
    ) -> Self {
        Self {
            camera,
            prediction_service,
            poll_interval_ms,
        }
    }

    pub async fn run(&self, mut shutdown_rx: broadcast::Receiver<()>) {
        let camera = self.camera.clone();
        let prediction_service = self.prediction_service.clone();
        let poll_interval_ms = self.poll_interval_ms;

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = Self::poll_and_predict(&camera, prediction_service.clone()) => {},
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
