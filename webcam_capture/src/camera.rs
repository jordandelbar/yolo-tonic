use opencv::{core, core::Mat, imgcodecs, imgproc, prelude::*, videoio};
use thiserror::Error;
use tokio::sync::Mutex;
use yolo_proto::BoundingBox;

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
}

impl From<opencv::Error> for CameraError {
    fn from(err: opencv::Error) -> Self {
        CameraError::OpenCvError(err)
    }
}

#[derive(Debug)]
pub struct Camera {
    pub capture: Mutex<videoio::VideoCapture>,
    pub predictions: Mutex<Vec<BoundingBox>>,
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

                imgproc::rectangle(
                    &mut frame,
                    core::Rect::new(x1, y1, x2 - x1, y2 - y1),
                    core::Scalar::new(0.0, 255.0, 0.0, 0.0),
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
                    core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                    1,
                    imgproc::LINE_AA,
                    false,
                )
                .map_err(CameraError::from)?;
            }
            let mut buf = opencv::core::Vector::<u8>::new();
            imgcodecs::imencode(".jpg", &frame, &mut buf, &opencv::core::Vector::new())
                .map_err(CameraError::EncodeFrameFailed)?;
            return Ok(Some(buf.into()));
        }
        Ok(None)
    }
}
