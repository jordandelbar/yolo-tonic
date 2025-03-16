use crate::prediction::{BoundingBoxWithLabels, PredictionService, PredictionServiceError};
use opencv::{
    core,
    core::{Mat, Vector},
    imgcodecs, imgproc,
    prelude::*,
    videoio,
};
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
    #[error("Failed to encode frame: {0}")]
    EncodeFrameFailed(opencv::Error),
    #[error("OpenCV error: {0}")]
    OpenCvError(opencv::Error),
    #[error("Prediction service error: {0}")]
    PredictionError(#[from] PredictionServiceError),
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
            for bbox in predictions.iter() {
                let x1 = bbox.x1 as i32;
                let y1 = bbox.y1 as i32;
                let x2 = bbox.x2 as i32;
                let y2 = bbox.y2 as i32;
                let label = format!("{}: {:.2}", bbox.class_label, bbox.confidence);

                // OpenCV uses BGR
                let color =
                    core::Scalar::new(bbox.blue as f64, bbox.green as f64, bbox.red as f64, 0.0);

                imgproc::rectangle(
                    &mut frame,
                    core::Rect::new(x1, y1, x2 - x1, y2 - y1),
                    color,
                    2,
                    imgproc::LINE_8,
                    0,
                )
                .map_err(CameraError::from)?;

                imgproc::put_text(
                    &mut frame,
                    &label,
                    core::Point::new(x1, y1 - 5),
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    imgproc::LINE_AA,
                    false,
                )
                .map_err(CameraError::from)?;
            }
            let mut buf = Vector::<u8>::new();
            imgcodecs::imencode(".jpg", &frame, &mut buf, &Vector::new())
                .map_err(CameraError::EncodeFrameFailed)?;
            return Ok(Some(buf.into()));
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
        camera: &Arc<crate::camera::Camera>,
        prediction_service: Arc<PredictionService>,
    ) -> Result<(), crate::camera::CameraError> {
        let frame = camera.capture_frame().await?;
        if frame.empty() {
            return Ok(());
        }

        let mut buf = opencv::core::Vector::<u8>::new();
        opencv::imgcodecs::imencode(".jpg", &frame, &mut buf, &opencv::core::Vector::new())
            .map_err(crate::camera::CameraError::EncodeFrameFailed)?;
        let image_data: Vec<u8> = buf.into();

        let detections = prediction_service.predict(image_data).await?;
        let mut pred_lock = camera.predictions.lock().await;
        *pred_lock = detections;

        Ok(())
    }
}
