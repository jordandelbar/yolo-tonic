use crate::ModelService;
use crate::{ImageFrame, PredictionBatch};
use image::{imageops::FilterType, GenericImageView};
use ndarray::{Array, Ix4};
use ort::session::{Session, SessionOutputs};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tonic::{async_trait, Status};

fn transform_image_frame(image_frame: &ImageFrame) -> Result<Array<f32, Ix4>, String> {
    let image_data = &image_frame.image_data;

    let image_reader = image::ImageReader::new(std::io::Cursor::new(image_data))
        .with_guessed_format()
        .map_err(|e| format!("Error decoding image: {}", e))?;

    let original_img = image_reader
        .decode()
        .map_err(|e| format!("Error decoding image: {}", e))?;

    let (_img_width, _img_height) = original_img.dimensions();
    let img = original_img.resize_exact(640, 640, FilterType::CatmullRom);

    let mut input = Array::zeros((1, 3, 640, 640));
    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2 .0;
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }

    Ok(input)
}

#[derive(Clone)]
pub struct OrtModelService {
    sessions: Arc<Vec<Arc<Session>>>,
    counter: Arc<AtomicUsize>,
}

impl OrtModelService {
    pub fn new(model_path: &str, num_instances: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let sessions = (0..num_instances)
            .map(|_| {
                let session = Session::builder()?.commit_from_file(model_path)?;
                Ok(Arc::new(session))
            })
            .collect::<Result<Vec<_>, ort::Error>>()?;
        Ok(Self {
            counter: Arc::new(AtomicUsize::new(0)),
            sessions: Arc::new(sessions),
        })
    }

    pub fn run_inference(&self, input: &Array<f32, Ix4>) -> Result<SessionOutputs, Status> {
        let index = self.counter.fetch_add(1, Ordering::SeqCst) % self.sessions.len();
        let session = &self.sessions[index];

        // TODO: improve error handling
        let output = session.run(ort::inputs![input.view()].unwrap()).unwrap();
        Ok(output)
    }
}

#[async_trait]
impl ModelService for OrtModelService {
    async fn predict(&self, frame: ImageFrame) -> Result<PredictionBatch, Status> {
        let input_result = transform_image_frame(&frame);
        let input = match input_result {
            Ok(input) => input,
            Err(err) => {
                return Err(Status::invalid_argument(format!(
                    "Image transformation error: {}",
                    err
                )))
            }
        };

        let outputs_result = self.run_inference(&input);
        let outputs = match outputs_result {
            Ok(outputs) => outputs,
            Err(err) => return Err(err),
        };

        // Placeholder
        let prediction_batch = PredictionBatch::default();
        Ok(prediction_batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};
    use ndarray::{Array, Ix4};
    use std::io::Cursor;

    #[test]
    fn test_transform_image_frame() {
        let img = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_pixel(100, 100, Rgb([255, 0, 0]));
        let mut image_data: Vec<u8> = Vec::new();
        let mut cursor = Cursor::new(&mut image_data);
        img.write_to(&mut cursor, image::ImageFormat::Png).unwrap();

        let image_frame = ImageFrame {
            image_data: cursor.get_ref().to_vec(),
            timestamp: 0,
        };

        let input_array_result = transform_image_frame(&image_frame);

        assert!(input_array_result.is_ok());

        let input_array: Array<f32, Ix4> = input_array_result.unwrap();

        assert_eq!(input_array.shape(), &[1, 3, 640, 640]);
    }
}
