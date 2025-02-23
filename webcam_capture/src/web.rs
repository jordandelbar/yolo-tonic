use crate::camera::Camera;
use crate::config::Settings;
use crate::prediction::{prediction_worker, PredictionClient};
use crate::routes::video_stream;

use axum::routing::{get, Router};
use std::{error::Error, sync::Arc};
use tokio::net::TcpListener;

pub async fn start_app(config: Settings) -> Result<(), Box<dyn Error>> {
    let camera = match Camera::new().await {
        Ok(cam) => Arc::new(cam),
        Err(e) => {
            tracing::error!("Failed to initialize camera: {:?}", e);
            return Err(Box::new(e));
        }
    };

    let camera_clone = camera.clone();
    let prediction_client = Arc::new(PredictionClient::new(
        config.prediction_service.get_address(),
    ));

    tokio::spawn(async move {
        loop {
            tracing::info!("Starting prediction worker...");
            prediction_worker(camera_clone.clone(), prediction_client.clone()).await;
            tracing::error!("Prediction worker exited. Restarting in 5 seconds...");
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    });

    let app = Router::new()
        .route("/video_feed", get(video_stream))
        .with_state(camera)
        .into_make_service();

    let addr = config.app.get_address();
    tracing::info!("starting app on {}", &addr);
    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
