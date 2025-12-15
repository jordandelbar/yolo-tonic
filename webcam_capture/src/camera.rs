use crate::{
    bounding_box::BoundingBoxWithLabels,
    config::CameraConfig,
    cv_utils::{CvImage, CvUtilsError},
    prediction::PredictionService,
    telemetry::Metrics,
};
use opencv::prelude::*;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use thiserror::Error;
use tokio::{
    sync::{broadcast, Mutex},
    task::JoinHandle,
    time::{sleep, Duration, Instant},
};
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
    ImageEncode(CvUtilsError),
}

pub struct Camera {
    device_id: i32,
    running: Arc<AtomicBool>,
    prediction_running: Arc<AtomicBool>,
    raw_frame_sender: broadcast::Sender<Vec<u8>>,
    frame_sender: broadcast::Sender<Vec<u8>>,
    prediction_service: Arc<PredictionService>,
    predictions_lock: Arc<Mutex<Vec<BoundingBoxWithLabels>>>,
    stream_delay: u64,
    prediction_delay: u64,
    fps_camera_frame_count: Arc<AtomicUsize>,
    fps_prediction_frame_count: Arc<AtomicUsize>,
    metrics: Arc<Metrics>,
}

impl Camera {
    pub fn new(
        device_id: i32,
        prediction_service: Arc<PredictionService>,
        camera_config: &CameraConfig,
        metrics: Arc<Metrics>,
    ) -> Result<Self, CameraError> {
        let (tx, _) = broadcast::channel(32);
        let (raw_tx, _) = broadcast::channel(16);

        let stream_delay = camera_config.get_stream_delay_ms();
        let prediction_delay = camera_config.get_prediction_delay_ms();
        tracing::info!(
            "Initialize camera with {} ms stream delay and {} ms prediction delay",
            stream_delay,
            prediction_delay
        );

        Ok(Self {
            device_id,
            running: Arc::new(AtomicBool::new(true)),
            prediction_running: Arc::new(AtomicBool::new(true)),
            raw_frame_sender: raw_tx,
            frame_sender: tx,
            prediction_service,
            predictions_lock: Arc::new(Mutex::new(vec![])),
            stream_delay,
            prediction_delay,
            fps_camera_frame_count: Arc::new(AtomicUsize::new(0)),
            fps_prediction_frame_count: Arc::new(AtomicUsize::new(0)),
            metrics,
        })
    }

    pub async fn start(
        &self,
    ) -> Result<
        (
            JoinHandle<Result<(), CameraError>>,
            JoinHandle<Result<(), CameraError>>,
        ),
        CameraError,
    > {
        let devide_id = self.device_id;
        let camera_running = self.running.clone();
        let prediction_running = self.prediction_running.clone();
        let raw_frame_sender = self.raw_frame_sender.clone();
        let frame_sender = self.frame_sender.clone();
        let prediction_service = self.prediction_service.clone();
        let predictions_lock1 = self.predictions_lock.clone();
        let predictions_lock2 = self.predictions_lock.clone();
        let stream_delay = self.stream_delay;
        let prediction_delay = self.prediction_delay;
        let metrics = self.metrics.clone();
        let fps_camera_frame_count = self.fps_camera_frame_count.clone();
        let fps_prediction_frame_count = self.fps_prediction_frame_count.clone();

        let frame_thread = tokio::spawn(async move {
            let mut camera =
                opencv::videoio::VideoCapture::new(devide_id, opencv::videoio::CAP_ANY)
                    .map_err(CameraError::OpenCamera)?;

            while camera_running.load(Ordering::Relaxed) {
                let mut image = CvImage::new();
                if camera.read(&mut image.mat).unwrap_or(false) {
                    if raw_frame_sender.send(image.to_jpg().unwrap()).is_err() {
                        tracing::warn!("No prediction receiver listening for raw frames.");
                    }

                    match process_frame(image, &predictions_lock1).await {
                        Ok(annotated_frame) => {
                            fps_camera_frame_count.fetch_add(1, Ordering::Relaxed);
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
                sleep(Duration::from_millis(stream_delay)).await;
            }
            drop(camera);

            Ok(())
        });

        let raw_frame_sender_clone = self.raw_frame_sender.clone();
        let mut raw_rx = raw_frame_sender_clone.subscribe();
        let prediction_thread = tokio::spawn(async move {
            while prediction_running.load(Ordering::Relaxed) {
                match raw_rx.recv().await {
                    Ok(frame_data) => {
                        let start = Instant::now();
                        match prediction_service.predict(frame_data).await {
                            Ok(predictions) => {
                                let elapsed = start.elapsed().as_millis();
                                metrics.record_prediction_duration(elapsed as u64, "camera");
                                fps_prediction_frame_count.fetch_add(1, Ordering::Relaxed);

                                let mut lock = predictions_lock2.lock().await;
                                *lock = predictions;
                            }
                            Err(e) => {
                                tracing::error!("Prediction error: {:?}", e);
                                return Err(CameraError::Prediction(
                                    "Error when trying to predict".to_string(),
                                ));
                            }
                        }
                        metrics.record_request("camera");
                    }
                    Err(broadcast::error::RecvError::Lagged(lagged)) => {
                        tracing::info!("Prediction thread lagged, dropped {} frames", lagged);
                        raw_rx = raw_frame_sender_clone.subscribe();
                        continue;
                    }
                    Err(_) => break,
                }
                sleep(Duration::from_millis(prediction_delay)).await;
            }
            Ok(())
        });

        let fps_camera_frame_count = self.fps_camera_frame_count.clone();
        let fps_prediction_frame_count = self.fps_prediction_frame_count.clone();
        let metrics = self.metrics.clone();

        let _ = tokio::spawn(async move {
            loop {
                sleep(Duration::from_secs(1)).await;
                let camera_count = fps_camera_frame_count.swap(0, Ordering::AcqRel);
                let prediction_count = fps_prediction_frame_count.swap(0, Ordering::AcqRel);

                // Since we sleep exactly 1 second, count equals FPS
                let camera_fps = camera_count as f64;
                let prediction_fps = prediction_count as f64;

                metrics.record_camera_fps(camera_fps, "camera");
                metrics.record_prediction_fps(prediction_fps, "camera");
            }
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
    mut image: CvImage,
    predictions_lock: &Arc<Mutex<Vec<BoundingBoxWithLabels>>>,
) -> Result<Vec<u8>, CameraError> {
    let predictions = predictions_lock.lock().await.clone();

    image
        .annotate(&predictions)
        .map_err(|e| CameraError::FrameProcessing(e.to_string()))?
        .to_jpg()
        .map_err(CameraError::ImageEncode)
}
