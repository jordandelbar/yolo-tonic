use crate::{
    config::{ModelConfig, Validatable},
    model_service::ModelService,
};
use image::{imageops::FilterType, GenericImageView};
use ndarray::{s, Array, Axis, Ix4};
use ort::{
    execution_providers::TensorRTExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use tonic::{async_trait, Status};
use yolo_proto::{BoundingBox, ImageFrame, PredictionBatch};

fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1))
        - intersection(box1, box2)
}

fn transform_image_frame(image_frame: &ImageFrame) -> Result<(Array<f32, Ix4>, u32, u32), String> {
    let image_data = &image_frame.image_data;

    let image_reader = image::ImageReader::new(std::io::Cursor::new(image_data))
        .with_guessed_format()
        .map_err(|e| format!("Error decoding image: {}", e))?;

    let original_img = image_reader
        .decode()
        .map_err(|e| format!("Error decoding image: {}", e))?;

    let (img_width, img_height) = original_img.dimensions();
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

    Ok((input, img_height, img_width))
}

#[derive(Clone)]
pub struct OrtModelService {
    sessions: Arc<Vec<Arc<Mutex<Session>>>>,
    counter: Arc<AtomicUsize>,
    min_probability: f32,
}

impl OrtModelService {
    pub fn new(model_config: &ModelConfig) -> Result<Self, Box<dyn std::error::Error>> {
        ort::init()
            .with_execution_providers([TensorRTExecutionProvider::default()
                .with_engine_cache(true)
                .build()])
            .commit()?;
        let num_instances = model_config.num_instances;
        let sessions = (0..num_instances)
            .map(|_| {
                let session = Session::builder()?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .commit_from_file(model_config.get_path())?;
                Ok(Arc::new(Mutex::new(session)))
            })
            .collect::<Result<Vec<_>, ort::Error>>()?;

        tracing::info!("Created {} ONNX sessions", num_instances);

        Ok(Self {
            counter: Arc::new(AtomicUsize::new(0)),
            sessions: Arc::new(sessions),
            min_probability: model_config.min_probability,
        })
    }

    pub fn run_inference(
        &self,
        input: &Array<f32, Ix4>,
    ) -> Result<ndarray::ArrayD<f32>, Box<Status>> {
        let index = self.counter.fetch_add(1, Ordering::SeqCst) % self.sessions.len();
        let session_arc = &self.sessions[index];
        let mut session = session_arc
            .lock()
            .map_err(|e| Status::internal(format!("session mutex poisoned: {}", e)))?;

        tracing::debug!("Handling request with session {}", index);
        let owned_buffer;
        let input_view = if input.view().is_standard_layout() {
            input.view()
        } else {
            owned_buffer = input.to_owned();
            owned_buffer.view()
        };

        let tensor_ref = TensorRef::from_array_view(input_view)
            .map_err(|e| Status::internal(format!("failed to build tensor: {}", e)))?;

        let input_tensor = ort::inputs![tensor_ref];

        let outputs = session
            .run(input_tensor)
            .map_err(|e| Status::internal(format!("inference failed: {}", e)))?;

        let (shape, data) = outputs["output0"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Status::internal(format!("failed to extract tensor: {}", e)))?;

        let ix = shape.to_ixdyn();
        let array = ndarray::ArrayD::from_shape_vec(ix, data.to_vec())
            .map_err(|e| Status::internal(format!("invalid tensor shape: {}", e)))?;

        Ok(array)
    }
}

#[async_trait]
impl ModelService for OrtModelService {
    async fn predict(&self, frame: ImageFrame) -> Result<PredictionBatch, Status> {
        let input_result = transform_image_frame(&frame);
        let (input, img_height, img_width) = match input_result {
            Ok(result) => result,
            Err(err) => {
                return Err(Status::invalid_argument(format!(
                    "Image transformation error: {}",
                    err
                )))
            }
        };

        let outputs_array = self.run_inference(&input);
        let outputs = match outputs_array {
            Ok(outputs) => outputs,
            Err(err) => return Err(*err),
        };

        let mut boxes = Vec::new();
        let output = outputs.slice(s![.., .., 0]);

        for row in output.axis_iter(Axis(0)) {
            let row: Vec<_> = row.iter().copied().collect();
            let (class_id, prob) = row
                .iter()
                .skip(4)
                .enumerate()
                .map(|(index, value)| (index, *value))
                .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
                .unwrap();

            if prob < self.min_probability {
                continue;
            }

            let xc = row[0] / 640. * (img_width as f32);
            let yc = row[1] / 640. * (img_height as f32);
            let w = row[2] / 640. * (img_width as f32);
            let h = row[3] / 640. * (img_height as f32);

            boxes.push(BoundingBox {
                class_id: class_id.try_into().unwrap(),
                confidence: prob,
                x1: xc - w / 2.,
                y1: yc - h / 2.,
                x2: xc + w / 2.,
                y2: yc + h / 2.,
            });
        }

        boxes.sort_by(|box1, box2| box2.confidence.total_cmp(&box1.confidence));
        let mut result = Vec::new();

        while !boxes.is_empty() {
            result.push(boxes[0]);
            boxes = boxes
                .iter()
                .filter(|box1| intersection(&boxes[0], box1) / union(&boxes[0], box1) < 0.7)
                .cloned()
                .collect();
        }

        let mut prediction_batch = PredictionBatch::default();
        for bbox in result {
            let prediction = BoundingBox {
                class_id: bbox.class_id,
                confidence: bbox.confidence,
                x1: bbox.x1,
                y1: bbox.y1,
                x2: bbox.x2,
                y2: bbox.y2,
            };
            prediction_batch.detections.push(prediction);
        }

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

        let (input_array, img_height, img_width): (Array<f32, Ix4>, u32, u32) =
            input_array_result.unwrap();

        assert_eq!(input_array.shape(), &[1, 3, 640, 640]);
        assert_eq!(img_width, 100);
        assert_eq!(img_height, 100);
    }
}
