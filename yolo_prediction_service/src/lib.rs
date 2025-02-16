mod app;
mod model_service;

use proto::yolo_service_server::{YoloService, YoloServiceServer};
use proto::{BoundingBox, ImageFrame, PredictionBatch};
use std::sync::Arc;
use tonic::async_trait;
use tonic::{Request, Response, Status};
mod proto {
    tonic::include_proto!("yolo_service");
}
pub use app::App;
pub use model_service::OrtModelService;

#[async_trait]
pub trait ModelService: Send + Sync + Clone + 'static {
    async fn predict(&self, frame: ImageFrame) -> Result<PredictionBatch, Status>;
}

#[derive(Debug, Clone)]
struct InferenceService<M: ModelService> {
    model_service: Arc<M>,
}

impl<M: ModelService> InferenceService<M> {
    pub fn new(model_service: M) -> Self {
        Self {
            model_service: Arc::new(model_service),
        }
    }
}

#[async_trait]
impl<M: ModelService> YoloService for InferenceService<M> {
    async fn predict(
        &self,
        request: Request<ImageFrame>,
    ) -> Result<Response<PredictionBatch>, Status> {
        let image_frame = request.into_inner();
        let model_service = self.model_service.clone();
        let batch = model_service.predict(image_frame).await?;

        Ok(Response::new(batch))
    }
}

#[derive(Clone)]
pub struct MockModelService {}

#[async_trait]
impl ModelService for MockModelService {
    async fn predict(&self, frame: ImageFrame) -> Result<PredictionBatch, Status> {
        let detections = vec![
            BoundingBox {
                x1: 10.0,
                y1: 20.0,
                x2: 100.0,
                y2: 150.0,
                class_label: "person".to_string(),
                confidence: 0.95,
            },
            BoundingBox {
                x1: 200.0,
                y1: 50.0,
                x2: 300.0,
                y2: 200.0,
                class_label: "bicycle".to_string(),
                confidence: 0.88,
            },
        ];

        Ok(PredictionBatch {
            detections,
            timestamp: frame.timestamp,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct MockModelService {}

    #[async_trait]
    impl ModelService for MockModelService {
        async fn predict(&self, frame: ImageFrame) -> Result<PredictionBatch, Status> {
            let detections = vec![
                BoundingBox {
                    x1: 10.0,
                    y1: 20.0,
                    x2: 100.0,
                    y2: 150.0,
                    class_label: "person".to_string(),
                    confidence: 0.95,
                },
                BoundingBox {
                    x1: 200.0,
                    y1: 50.0,
                    x2: 300.0,
                    y2: 200.0,
                    class_label: "bicycle".to_string(),
                    confidence: 0.88,
                },
            ];

            Ok(PredictionBatch {
                detections,
                timestamp: frame.timestamp,
            })
        }
    }

    #[tokio::test]
    async fn test_predict() -> Result<(), Box<dyn std::error::Error>> {
        let mock_model = MockModelService {};
        let inference_service = InferenceService::new(mock_model);

        let image_frame = ImageFrame {
            image_data: vec![0; 100],
            timestamp: 12345,
        };

        let request = Request::new(image_frame);
        let response = inference_service.predict(request).await?;

        let batch = response.into_inner();
        assert_eq!(batch.detections.len(), 2);
        assert_eq!(batch.detections[0].class_label, "person");
        assert_eq!(batch.detections[1].class_label, "bicycle");

        Ok(())
    }
}
