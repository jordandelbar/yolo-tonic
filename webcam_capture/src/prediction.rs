use crate::bounding_box::BoundingBoxWithLabels;
use crate::config::PredictionServiceConfig;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::{
    sync::Mutex,
    time::{sleep, timeout, Duration},
};
use tonic::{
    transport::{Channel, Error},
    Request, Status,
};
use tracing::instrument;
use yolo_proto::{
    yolo_service_client::YoloServiceClient, BoundingBox, ColorLabel, Empty, ImageFrame,
};

#[derive(Error, Debug)]
pub enum PredictionServiceError {
    #[error("Failed to connect to gRPC server: {0}")]
    ConnectionFailed(#[from] Error),
    #[error("Maximum connection retries exceeded.")]
    MaxRetriesExceeded,
    #[error("gRPC request failed: {0}")]
    GrpcRequestFailed(#[from] Status),
}

pub struct PredictionService {
    client: Mutex<YoloServiceClient<Channel>>,
    class_labels: Mutex<Vec<ColorLabel>>,
}

impl PredictionService {
    pub async fn new(
        prediction_config: &PredictionServiceConfig,
    ) -> Result<Self, PredictionServiceError> {
        let client = Self::get_client(prediction_config.get_address()).await?;

        let service = Self {
            client: Mutex::new(client),
            class_labels: Mutex::new(Vec::new()),
        };

        // We need the client to initialize the labels
        {
            let mut client = service.client.lock().await;
            let request = Request::new(Empty {});
            let response = client.get_yolo_class_labels(request).await?;
            let labels = response.into_inner();

            drop(client);

            let mut class_labels = service.class_labels.lock().await;
            *class_labels = labels.class_labels;
        }

        Ok(service)
    }

    async fn get_client(
        address: String,
    ) -> Result<YoloServiceClient<Channel>, PredictionServiceError> {
        let mut retry_delay = Duration::from_millis(50);
        let max_retry_delay = Duration::from_secs(1);
        let max_retries = 10;
        let mut retry_count = 0;

        while retry_count < max_retries {
            match timeout(
                Duration::from_secs(1),
                YoloServiceClient::connect(address.clone()),
            )
            .await
            {
                Ok(Ok(client)) => return Ok(client),
                Ok(Err(e)) => {
                    tracing::error!("Failed to connect to gRPC server: {:?}", e);
                }
                Err(_) => {
                    tracing::error!("Connection timeout");
                }
            }

            retry_count += 1;
            let jitter = rand::random::<f32>() * 0.2 + 0.9;
            sleep(retry_delay.mul_f32(jitter)).await;
            retry_delay = (retry_delay * 2).min(max_retry_delay);
        }

        Err(PredictionServiceError::MaxRetriesExceeded)
    }

    #[instrument(skip(self, image_data))]
    pub async fn predict(
        &self,
        image_data: Vec<u8>,
    ) -> Result<Vec<BoundingBoxWithLabels>, PredictionServiceError> {
        let mut client = self.client.lock().await;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let request = Request::new(ImageFrame {
            image_data,
            timestamp,
        });

        let response = client.predict(request).await?;
        let detections = response.into_inner().detections;
        let class_labels = self.class_labels.lock().await;

        let labeled_detections: Vec<BoundingBoxWithLabels> = detections
            .into_iter()
            .map(|bbox: BoundingBox| {
                if let Some(color_label) = class_labels.get(bbox.class_id as usize) {
                    BoundingBoxWithLabels {
                        x1: bbox.x1,
                        y1: bbox.y1,
                        x2: bbox.x2,
                        y2: bbox.y2,
                        class_label: color_label.label.clone(),
                        red: color_label.red,
                        green: color_label.green,
                        blue: color_label.blue,
                        confidence: bbox.confidence,
                    }
                } else {
                    BoundingBoxWithLabels {
                        x1: bbox.x1,
                        y1: bbox.y1,
                        x2: bbox.x2,
                        y2: bbox.y2,
                        class_label: format!("Unknown class {}", bbox.class_id),
                        red: 0,
                        green: 0,
                        blue: 0,
                        confidence: bbox.confidence,
                    }
                }
            })
            .collect();

        Ok(labeled_detections)
    }
}
