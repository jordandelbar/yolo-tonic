use crate::camera::Camera;
use crate::config::Settings;
use crate::prediction::{prediction_worker, PredictionClient};
use crate::routes::video_feed;
use crate::stream::VideoStream;

use axum::routing::{get, Router};
use std::{error::Error, sync::Arc};
use tokio::{net::TcpListener, signal};

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
    let video_stream = VideoStream::new(camera, config.app.get_stream_delay_ms());

    let prediction_handle = tokio::spawn(async move {
        loop {
            tracing::info!("Starting prediction worker...");
            prediction_worker(
                camera_clone.clone(),
                prediction_client.clone(),
                config.prediction_service.get_prediction_delay_ms(),
            )
            .await;
            tracing::error!("Prediction worker exited. Restarting in 5 seconds...");
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    });

    let app = Router::new()
        .route("/video_feed", get(video_feed))
        .with_state(video_stream)
        .into_make_service();

    let addr = config.app.get_address();
    tracing::info!("Starting app on {}", &addr);
    let listener = TcpListener::bind(&addr).await?;

    let server = axum::serve(listener, app);

    let graceful_shutdown = async move {
        shutdown_signal().await;
        tracing::info!("Signal received, starting graceful shutdown");
        prediction_handle.abort();
    };

    tokio::select! {
        result = server.with_graceful_shutdown(graceful_shutdown) => {
            result?;
        }
        _ = signal::ctrl_c() => {
            tracing::info!{"Forcing shutdown after Ctrl-C"}
        }
    }

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
