use crate::bounding_box::BoundingBoxWithLabels;
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
}

impl From<opencv::Error> for CvUtilsError {
    fn from(err: opencv::Error) -> Self {
        CvUtilsError::OpenCvError(err)
    }
}

pub struct ImageConverter;

impl ImageConverter {
    pub fn encode_mat_to_jpg(mat: &Mat) -> Result<Vec<u8>, CvUtilsError> {
        let mut buf = Vector::<u8>::new();
        imgcodecs::imencode(".jpg", mat, &mut buf, &Vector::new())
            .map_err(CvUtilsError::EncodeFrameFailed)?;
        Ok(buf.into())
    }

    pub fn annotate_frame(
        frame: &mut Mat,
        bboxes: &[BoundingBoxWithLabels],
    ) -> Result<(), CvUtilsError> {
        for bbox in bboxes {
            let x1 = bbox.x1 as i32;
            let y1 = bbox.y1 as i32;
            let x2 = bbox.x2 as i32;
            let y2 = bbox.y2 as i32;
            let label = format!("{}: {:.2}", bbox.class_label, bbox.confidence);

            let color = Scalar::new(bbox.blue as f64, bbox.green as f64, bbox.red as f64, 0.0);

            imgproc::rectangle(
                frame,
                Rect::new(x1, y1, x2 - x1, y2 - y1),
                color,
                2,
                imgproc::LINE_8,
                0,
            )
            .map_err(CvUtilsError::from)?;

            imgproc::put_text(
                frame,
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
        Ok(())
    }
}
