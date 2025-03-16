use crate::bounding_box::BoundingBoxWithLabels;
use crate::camera::{Camera, CameraError};
use crate::config::{PredictionPollingConfig, PredictionServiceConfig};
use crate::cv_utils::{CvUtilsError, ImageConverter};
use opencv::prelude::*;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::{
    sync::{broadcast, Mutex},
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
    #[error("Cv utils error: {0}")]
    OpenCvUtilsError(#[from] CvUtilsError),
    #[error("Camera error: {0}")]
    PredictionCameraError(#[from] CameraError),
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

pub struct PredictionPoller {
    camera: Arc<Camera>,
    prediction_service: Arc<PredictionService>,
    poll_interval_ms: u64,
    max_retries: u64,
    initial_delay: u64,
    backoff_factor: u32,
    max_consecutive_failures: u64,
}

impl PredictionPoller {
    pub fn new(
        camera: Arc<Camera>,
        prediction_service: Arc<PredictionService>,
        prediction_polling_config: &PredictionPollingConfig,
    ) -> Self {
        Self {
            camera,
            prediction_service,
            poll_interval_ms: prediction_polling_config.get_prediction_delay_ms(),
            max_retries: prediction_polling_config.max_retries,
            initial_delay: prediction_polling_config.initial_delay,
            backoff_factor: prediction_polling_config.backoff_factor,
            max_consecutive_failures: prediction_polling_config.max_consecutive_failures,
        }
    }

    pub async fn run(&self, mut shutdown_rx: broadcast::Receiver<()>) {
        let camera = self.camera.clone();
        let prediction_service = self.prediction_service.clone();
        let poll_interval_ms = self.poll_interval_ms;
        let max_retries = self.max_retries;
        let initial_delay = Duration::from_millis(self.initial_delay);
        let backoff_factor = self.backoff_factor;
        let mut consecutive_failures = 0;
        let max_consecutive_failures = self.max_consecutive_failures;

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    result = Self::poll_and_predict(&camera, prediction_service.clone()) => {
                        match result {
                            Ok(_) => {
                                consecutive_failures = 0;
                            },
                            Err(ref err) => {
                                tracing::error!("Error during polling: {:?}", err);
                                let mut retry_delay = initial_delay;
                                let mut retry_successful = false;
                                for retry_count in 0..max_retries {
                                    tracing::warn!(
                                        "Retrying poll (attempt {}/{}) on consecutive failures: {}/{}",
                                        retry_count + 1,
                                        max_retries,
                                        consecutive_failures,
                                        max_consecutive_failures
                                    );
                                    sleep(retry_delay).await;

                                    let retry_result = Self::poll_and_predict(&camera, prediction_service.clone()).await;
                                    if retry_result.is_ok() {
                                        tracing::info!("Retry successful");
                                        retry_successful = true;
                                        break;
                                    } else {
                                        tracing::error!("Retry failed: {:?}", retry_result);
                                        retry_delay *= backoff_factor;
                                    }
                                }
                                if !retry_successful {
                                    consecutive_failures += 1;
                                    tracing::error!("Max number of retries reached, skipping current poll interval");
                                }
                                if consecutive_failures >= max_consecutive_failures {
                                    tracing::error!("Persistent failure detected. Exiting polling loop");
                                    break;
                                }
                            }
                        }
                    },
                    _ = shutdown_rx.recv() => {
                        tracing::info!("Camera polling received shutdown signal");
                        break;
                    }
                }

                sleep(Duration::from_millis(poll_interval_ms)).await;
            }
            tracing::info!("Camera polling stopped");
        });
    }

    async fn poll_and_predict(
        camera: &Arc<Camera>,
        prediction_service: Arc<PredictionService>,
    ) -> Result<(), PredictionServiceError> {
        let frame = camera.capture_frame().await?;
        if frame.empty() {
            return Ok(());
        }

        let image_data = ImageConverter::encode_mat_to_jpg(&frame)?;

        let detections = prediction_service.predict(image_data).await?;
        let mut pred_lock = camera.predictions.lock().await;
        *pred_lock = detections;

        Ok(())
    }
}
