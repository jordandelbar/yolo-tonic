use crate::camera::Camera;
use crate::config::PredictionServiceSettings;
use opencv::{core, core::Mat, imgcodecs, prelude::*};
use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::time::sleep;
use tracing::instrument;
use yolo_proto::{yolo_service_client::YoloServiceClient, ImageFrame};

#[instrument(skip(camera, prediction_config))]
pub async fn prediction_worker(camera: Arc<Camera>, prediction_config: PredictionServiceSettings) {
    let prediction_service_address = prediction_config.get_address();
    let mut client = YoloServiceClient::connect(prediction_service_address)
        .await
        .expect("failed to connect to gRPC server");

    loop {
        sleep(Duration::from_millis(60)).await;

        let frame = {
            let mut cam = camera.capture.lock().await;
            let mut frame = Mat::default();
            if cam.read(&mut frame).unwrap() && !frame.empty() {
                let mut buf = core::Vector::<u8>::new();
                imgcodecs::imencode(".jpg", &frame, &mut buf, &core::Vector::new())
                    .ok()
                    .unwrap();
                buf.into()
            } else {
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
                eprintln!("gRPC request failed: {:?}", e);
            }
        }
    }
}
