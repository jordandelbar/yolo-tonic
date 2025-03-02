use crate::camera::Camera;
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
use yolo_proto::{yolo_service_client::YoloServiceClient, ImageFrame};

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
    camera: Arc<Camera>,
    client: YoloServiceClient<Channel>,
    prediction_delay: u64,
}

impl PredictionService {
    pub async fn new(
        camera: Arc<Camera>,
        address: String,
        prediction_delay: u64,
    ) -> Result<Self, PredictionServiceError> {
        let client = Self::get_client(address).await?;
        Ok(Self {
            camera,
            client,
            prediction_delay,
        })
    }

    async fn get_client(
        address: String,
    ) -> Result<YoloServiceClient<Channel>, PredictionServiceError> {
        let mut retry_delay = Duration::from_millis(50);
        let max_retry_delay = Duration::from_secs(1);
        let max_retries = 5;
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
                    return Err(PredictionServiceError::ConnectionFailed(e));
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
                    let predictions = response.into_inner().detections;
                    let mut pred_lock = self.camera.predictions.lock().await;
                    *pred_lock = predictions;
                }
                Err(e) => {
                    tracing::error!("gRPC request failed: {:?}", e);
                }
            }
        }
    }
}
