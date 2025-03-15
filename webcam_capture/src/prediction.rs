use crate::{camera::Camera, config::PredictionServiceConfig};
use opencv::{core, imgcodecs, prelude::*};
use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;
use tokio::time::{sleep, timeout};
use tonic::{
    transport::{Channel, Error},
    Request, Status,
};
use tracing::instrument;
use yolo_proto::{
    yolo_service_client::YoloServiceClient, BoundingBox, ColorLabel, Empty, ImageFrame,
    YoloClassLabels,
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

#[derive(Debug)]
pub struct BoundingBoxWithLabels {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub class_label: String,
    pub red: u32,
    pub green: u32,
    pub blue: u32,
    pub confidence: f32,
}

pub struct PredictionService {
    camera: Arc<Camera>,
    client: YoloServiceClient<Channel>,
    prediction_delay: u64,
    class_labels: Vec<ColorLabel>,
}

impl PredictionService {
    pub async fn new(
        camera: Arc<Camera>,
        prediction_config: &PredictionServiceConfig,
    ) -> Result<Self, PredictionServiceError> {
        let client = Self::get_client(prediction_config.get_address()).await?;
        let mut service = Self {
            camera,
            client,
            prediction_delay: prediction_config.get_prediction_delay_ms(),
            class_labels: Vec::new(),
        };
        let labels = service.get_labels().await?;
        service.class_labels = labels.class_labels;
        Ok(service)
    }

    async fn get_client(
        address: String,
    ) -> Result<YoloServiceClient<Channel>, PredictionServiceError> {
        let mut retry_delay = Duration::from_millis(50);
        let max_retry_delay = Duration::from_secs(1);
        let max_retries = 10;
        let mut retry_count = 0;

        loop {
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

            if retry_count >= max_retries {
                return Err(PredictionServiceError::MaxRetriesExceeded);
            }

            sleep(retry_delay).await;
            retry_delay = (retry_delay * 2).min(max_retry_delay);
            retry_count += 1;
        }
    }

    pub async fn get_labels(&mut self) -> Result<YoloClassLabels, PredictionServiceError> {
        let request = Request::new(Empty {});
        match self.client.get_yolo_class_labels(request).await {
            Ok(response) => Ok(response.into_inner()),
            Err(e) => Err(PredictionServiceError::GrpcRequestFailed(e)),
        }
    }

    #[instrument(skip(self))]
    pub async fn run(&mut self) {
        loop {
            sleep(Duration::from_millis(self.prediction_delay)).await;

            let frame = match self.camera.capture_frame().await {
                Ok(frame) if !frame.empty() => {
                    let mut buf = core::Vector::<u8>::new();
                    if let Err(e) =
                        imgcodecs::imencode(".jpg", &frame, &mut buf, &core::Vector::new())
                    {
                        tracing::error!("Failed to encode frame: {:?}", e);
                        continue;
                    }
                    buf.into()
                }
                Ok(_) => {
                    continue;
                }
                Err(e) => {
                    tracing::error!("Failed to read frame: {:?}", e);
                    continue;
                }
            };

            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as i64;

            let request = Request::new(ImageFrame {
                image_data: frame,
                timestamp,
            });

            match self.client.predict(request).await {
                Ok(response) => {
                    let detections = response.into_inner().detections;
                    let mut pred_lock = self.camera.predictions.lock().await;
                    let labeled_detections: Vec<BoundingBoxWithLabels> = detections
                        .into_iter()
                        .map(|bbox: BoundingBox| {
                            if let Some(color_label) = self.class_labels.get(bbox.class_id as usize)
                            {
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
                    *pred_lock = labeled_detections;
                }
                Err(e) => {
                    tracing::error!("gRPC request failed: {:?}", e);
                }
            }
        }
    }
}
