use opencv::{core, core::Mat, imgcodecs, imgproc, prelude::*, videoio};
use tokio::sync::Mutex;
use yolo_proto::BoundingBox;

#[derive(Debug)]
pub struct Camera {
    pub capture: Mutex<videoio::VideoCapture>,
    pub predictions: Mutex<Vec<BoundingBox>>,
}

impl Camera {
    pub async fn new() -> Self {
        let capture = videoio::VideoCapture::new(0, videoio::CAP_ANY).unwrap();
        Self {
            capture: Mutex::new(capture),
            predictions: Mutex::new(vec![]),
        }
    }

    pub async fn get_frame(&self) -> Option<Vec<u8>> {
        let mut cam = self.capture.lock().await;
        let mut frame = Mat::default();
        if cam.read(&mut frame).unwrap() && !frame.empty() {
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
                .unwrap();

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
                .unwrap();
            }
            let mut buf = opencv::core::Vector::<u8>::new();
            imgcodecs::imencode(".jpg", &frame, &mut buf, &opencv::core::Vector::new()).ok()?;
            return Some(buf.into());
        }
        None
    }
}
