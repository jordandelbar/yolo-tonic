use crate::camera::Camera;
use opencv::{core, imgcodecs, prelude::*};
use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;
use tokio::time::{sleep, timeout};
use tracing::instrument;
use yolo_proto::{yolo_service_client::YoloServiceClient, ImageFrame};

#[derive(Error, Debug)]
pub enum PredictionClientError {
    #[error("Failed to connect to gRPC server: {0}")]
    ConnectionFailed(#[from] tonic::transport::Error),
    #[error("Maximum connection retries exceeded.")]
    MaxRetriesExceeded,
    #[error("gRPC request failed: {0}")]
    GrpcRequestFailed(#[from] tonic::Status),
}

pub struct PredictionClient {
    address: String,
}

impl PredictionClient {
    pub fn new(address: String) -> Self {
        PredictionClient { address }
    }

    pub async fn get_client(
        &self,
    ) -> Result<YoloServiceClient<tonic::transport::Channel>, PredictionClientError> {
        let mut retry_delay = Duration::from_millis(50);
        let max_retry_delay = Duration::from_secs(1);
        let max_retries = 5;
        let mut retry_count = 0;

        loop {
            match timeout(
                Duration::from_secs(1),
                YoloServiceClient::connect(self.address.clone()),
            )
            .await
            {
                Ok(Ok(client)) => return Ok(client),
                Ok(Err(e)) => {
                    tracing::error!("Failed to connect to gRPC server: {:?}", e);
                    return Err(PredictionClientError::ConnectionFailed(e));
                }
                Err(_) => {
                    tracing::error!("Connection timeout");
                }
            }

            if retry_count >= max_retries {
                return Err(PredictionClientError::MaxRetriesExceeded);
            }

            sleep(retry_delay).await;
            retry_delay = (retry_delay * 2).min(max_retry_delay);
            retry_count += 1;
        }
    }
}

#[instrument(skip(camera, prediction_client))]
pub async fn prediction_worker(camera: Arc<Camera>, prediction_client: Arc<PredictionClient>) {
    let mut client = match prediction_client.get_client().await {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Failed to get gRPC client: {:?}", e);
            return;
        }
    };
    loop {
        sleep(Duration::from_millis(60)).await;

        let frame = match camera.capture_frame().await {
            Ok(frame) if !frame.empty() => {
                let mut buf = core::Vector::<u8>::new();
                if let Err(e) = imgcodecs::imencode(".jpg", &frame, &mut buf, &core::Vector::new())
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

        let request = tonic::Request::new(ImageFrame {
            image_data: frame,
            timestamp,
        });

        match client.predict(request).await {
            Ok(response) => {
                let predictions = response.into_inner().detections;
                let mut pred_lock = camera.predictions.lock().await;
                *pred_lock = predictions;
            }
            Err(e) => {
                tracing::error!("gRPC request failed: {:?}", e);
            }
        }
    }
}
