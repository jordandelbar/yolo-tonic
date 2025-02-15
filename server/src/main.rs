use async_stream::stream;
use futures::Stream;
use proto::yolo_service_server::{YoloService, YoloServiceServer};
use proto::{BoundingBox, ImageFrame, PredictionBatch};
use std::pin::Pin;
use std::sync::Arc;
use tonic::async_trait;
use tonic::transport::Server;
use tonic::{Request, Response, Status};
mod proto {
    tonic::include_proto!("yolo_service");
}

#[async_trait]
pub trait ModelService: Send + Sync + 'static {
    async fn predict(&self, frame: ImageFrame) -> Result<PredictionBatch, Status>;
}

#[derive(Debug)]
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
    type PredictStreamStream = Pin<Box<dyn Stream<Item = Result<PredictionBatch, Status>> + Send>>;

    async fn predict_stream(
        &self,
        request: Request<tonic::Streaming<ImageFrame>>,
    ) -> Result<Response<Self::PredictStreamStream>, Status> {
        let mut stream = request.into_inner();
        let model_service = self.model_service.clone();

        let output_stream = stream! {
            // Move the value of the cloned Arc into the stream
            let model_service = model_service;
            while let Some(frame) = stream.message().await.transpose() {
                let model_service = model_service.clone();
                match frame {
                    Ok(image_frame) => {
                        let prediction_result = model_service.predict(image_frame).await;
                        match prediction_result {
                            Ok(batch) => yield Ok(batch),
                            Err(status) => yield Err(status),
                        }
                    }
                    Err(status) => {
                        yield Err(status);
                        break
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(output_stream)))
    }

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse().unwrap();
    let mock_model_service = MockModelService {};
    let inferer = InferenceService::new(mock_model_service);

    println!("Inference service listening on {}", addr);

    Server::builder()
        .add_service(YoloServiceServer::new(inferer))
        .serve(addr)
        .await?;

    Ok(())
}
