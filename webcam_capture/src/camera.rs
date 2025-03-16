use crate::bounding_box::BoundingBoxWithLabels;
use crate::cv_utils::{CvUtilsError, ImageConverter};
use opencv::{core::Mat, prelude::*, videoio};
use thiserror::Error;
use tokio::sync::Mutex;

#[derive(Error, Debug)]
pub enum CameraError {
    #[error("Failed to open camera: {0}")]
    OpenCameraFailed(opencv::Error),
    #[error("Failed to read frame: {0}")]
    ReadFrameFailed(opencv::Error),
    #[error("OpenCV error: {0}")]
    OpenCvError(opencv::Error),
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
