use crate::{
    bounding_box::BoundingBoxWithLabels, cv_utils::ImageConverter, prediction::PredictionService,
};
use opencv::prelude::*;
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use thiserror::Error;
use tokio::sync::{broadcast, Mutex};
use tokio_stream::wrappers::BroadcastStream;

#[derive(Error, Debug)]
pub enum CameraError {
    #[error("Failed to open camera: {0}")]
    OpenCamera(opencv::Error),
    #[error("Frame processing error: {0}")]
    FrameProcessing(String),
    #[error("Prediction error: {0}")]
    Prediction(String),
    #[error("Image encode error: {0}")]
    ImageEncode(crate::cv_utils::CvUtilsError),
}

pub struct Camera {
    device_id: i32,
    running: Arc<AtomicBool>,
    prediction_running: Arc<AtomicBool>,
    frame_sender: broadcast::Sender<Vec<u8>>,
    prediction_service: Arc<PredictionService>,
    predictions_lock: Arc<Mutex<Vec<BoundingBoxWithLabels>>>,
}

impl Camera {
    pub fn new(
        device_id: i32,
        prediction_service: Arc<PredictionService>,
    ) -> Result<Self, CameraError> {
        let (tx, _) = broadcast::channel(32);
        Ok(Self {
            device_id,
            running: Arc::new(AtomicBool::new(true)),
            prediction_running: Arc::new(AtomicBool::new(true)),
            frame_sender: tx,
            prediction_service,
            predictions_lock: Arc::new(Mutex::new(vec![])),
        })
    }

    pub async fn start(
        &self,
    ) -> Result<
        (
            tokio::task::JoinHandle<Result<(), CameraError>>,
            tokio::task::JoinHandle<Result<(), CameraError>>,
        ),
        CameraError,
    > {
        let devide_id = self.device_id;
        let camera_running = self.running.clone();
        let prediction_running = self.prediction_running.clone();
        let frame_sender = self.frame_sender.clone();
        let prediction_service = self.prediction_service.clone();
        let predictions_lock1 = self.predictions_lock.clone();
        let predictions_lock2 = self.predictions_lock.clone();

        let frame_thread = tokio::spawn(async move {
            let mut camera =
                opencv::videoio::VideoCapture::new(devide_id, opencv::videoio::CAP_ANY)
                    .map_err(CameraError::OpenCamera)?;

            while camera_running.load(Ordering::Relaxed) {
                let mut frame = Mat::default();
                if camera.read(&mut frame).unwrap_or(false) {
                    match process_frame(&frame, &predictions_lock1).await {
                        Ok(annotated_frame) => {
                            if frame_sender.send(annotated_frame).is_err() {
                                tracing::error!("Failed to send frame to prediction thread");
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::error!("Frame processing error: {:?}", e);
                            break;
                        }
                    }
                }
                tokio::time::sleep(Duration::from_millis(33)).await;
            }
            drop(camera);

            Ok(())
        });

        let frame_sender_clone = self.frame_sender.clone();
        let prediction_thread = tokio::spawn(async move {
            let mut rx = frame_sender_clone.subscribe();
            while prediction_running.load(Ordering::Relaxed) {
                match rx.recv().await {
                    Ok(frame_data) => match prediction_service.predict(frame_data).await {
                        Ok(predictions) => {
                            let mut lock = predictions_lock2.lock().await;
                            *lock = predictions;
                        }
                        Err(e) => {
                            tracing::error!("Prediction error: {:?}", e);
                            return Err(CameraError::Prediction(
                                "Error when trying to predict".to_string(),
                            ));
                        }
                    },
                    Err(broadcast::error::RecvError::Lagged(lagged)) => {
                        tracing::warn!("Prediction thread lagged, dropped {} frames", lagged);
                        rx = frame_sender_clone.subscribe();
                        continue;
                    }
                    Err(_) => break,
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            Ok(())
        });

        Ok((frame_thread, prediction_thread))
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
        self.prediction_running.store(false, Ordering::Relaxed);
    }

    pub fn subscribe(&self) -> BroadcastStream<Vec<u8>> {
        BroadcastStream::new(self.frame_sender.subscribe())
    }
}

async fn process_frame(
    frame: &Mat,
    predictions_lock: &Arc<Mutex<Vec<BoundingBoxWithLabels>>>,
) -> Result<Vec<u8>, CameraError> {
    let mut annotated_frame = frame.clone();
    let preds = predictions_lock.lock().await.clone();

    ImageConverter::annotate_frame(&mut annotated_frame, &preds)
        .map_err(|e| CameraError::FrameProcessing(e.to_string()))?;

    ImageConverter::encode_mat_to_jpg(&annotated_frame).map_err(CameraError::ImageEncode)
}
