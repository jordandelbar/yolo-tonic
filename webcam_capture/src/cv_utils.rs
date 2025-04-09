use crate::bounding_box::BoundingBoxWithLabels;
use axum::body::Bytes;
use opencv::{
    core::{Mat, Point, Rect, Scalar, Vector},
    imgcodecs, imgproc,
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CvUtilsError {
    #[error("Failed to encode frame: {0}")]
    EncodeFrameFailed(opencv::Error),
    #[error("OpenCV error: {0}")]
    OpenCvError(opencv::Error),
    #[error("OpenCV decode error: {0}")]
    OpenCvDecodeError(opencv::Error),
}

impl From<opencv::Error> for CvUtilsError {
    fn from(err: opencv::Error) -> Self {
        CvUtilsError::OpenCvError(err)
    }
}

pub struct CvImage {
    pub mat: Mat,
}

impl CvImage {
    pub fn new() -> Self {
        let mat = Mat::default();
        Self { mat }
    }

    pub fn from_bytes(bytes: Bytes) -> Result<Self, CvUtilsError> {
        let mat = imgcodecs::imdecode(&Vector::from_slice(&bytes), imgcodecs::IMREAD_COLOR)
            .map_err(CvUtilsError::OpenCvDecodeError)?;
        Ok(Self { mat })
    }

    pub fn to_jpg(&self) -> Result<Vec<u8>, CvUtilsError> {
        let mut buf = Vector::<u8>::new();
        imgcodecs::imencode(".jpg", &self.mat, &mut buf, &Vector::new())
            .map_err(CvUtilsError::EncodeFrameFailed)?;
        Ok(buf.into())
    }

    pub fn annotate(
        &mut self,
        bboxes: &[BoundingBoxWithLabels],
    ) -> Result<&mut Self, CvUtilsError> {
        for bbox in bboxes {
            let x1 = bbox.x1 as i32;
            let y1 = bbox.y1 as i32;
            let x2 = bbox.x2 as i32;
            let y2 = bbox.y2 as i32;
            let label = format!("{}: {:.2}", bbox.class_label, bbox.confidence);

            let color = Scalar::new(bbox.blue as f64, bbox.green as f64, bbox.red as f64, 0.0);

            imgproc::rectangle(
                &mut self.mat,
                Rect::new(x1, y1, x2 - x1, y2 - y1),
                color,
                2,
                imgproc::LINE_8,
                0,
            )
            .map_err(CvUtilsError::from)?;

            imgproc::put_text(
                &mut self.mat,
                &label,
                Point::new(x1, y1 - 5),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                imgproc::LINE_AA,
                false,
            )
            .map_err(CvUtilsError::from)?;
        }
        Ok(self)
    }
}
