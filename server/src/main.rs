use async_stream::stream;
use futures::Stream;
use proto::yolo_service_server::{YoloService, YoloServiceServer};
use proto::{BoundingBox, ImageFrame, PredictionBatch};
use std::pin::Pin;
use tonic::async_trait;
use tonic::transport::Server;
use tonic::{Request, Response, Status};
mod proto {
    tonic::include_proto!("yolo_service");
}
use image::{DynamicImage, ImageError};
use std::io::Cursor;

#[derive(Debug, Default)]
struct InferenceService {}

#[async_trait]
impl YoloService for InferenceService {
    type PredictStreamStream = Pin<Box<dyn Stream<Item = Result<PredictionBatch, Status>> + Send>>;

    async fn predict_stream(
        &self,
        request: Request<tonic::Streaming<ImageFrame>>,
    ) -> Result<Response<Self::PredictStreamStream>, Status> {
        let mut stream = request.into_inner();

        let output_stream = stream! {
            while let Some(frame) = stream.message().await.transpose() {
                match frame {
                    Ok(image_frame) => {
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

                        let batch = PredictionBatch {
                            detections,
                            timestamp: image_frame.timestamp,
                        };

                        yield Ok(batch);
                    }
                    Err(status) => {
                        yield Err(status);
                        break;
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
        let image_data = image_frame.image_data;

        let image_data_slice = image_data.as_slice();

        let image_reader = image::ImageReader::new(Cursor::new(image_data_slice))
            .with_guessed_format()
            .map_err(|e| Status::internal(e.to_string()))?;
        let image_result: Result<DynamicImage, ImageError> = image_reader.decode();

        match image_result {
            Ok(_img) => {
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

                let batch = PredictionBatch {
                    detections,
                    timestamp: image_frame.timestamp,
                };

                Ok(Response::new(batch))
            }
            Err(err) => {
                eprintln!("Error decoding image: {}", err);
                Err(Status::invalid_argument("Failed to decode image"))
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse().unwrap();
    let inferer = InferenceService::default();

    println!("Inference service listening on {}", addr);

    Server::builder()
        .add_service(YoloServiceServer::new(inferer))
        .serve(addr)
        .await?;

    Ok(())
}
