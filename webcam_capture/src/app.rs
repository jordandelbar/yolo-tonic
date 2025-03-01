use crate::camera::Camera;
use crate::config::Settings;
use crate::prediction::{prediction_worker, PredictionClient};
use crate::server::HttpServer;

use std::{error::Error, sync::Arc};
use tokio::{signal, sync::broadcast};

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

    let server = HttpServer::new(camera, config.clone()).await?;

    let (shutdown_tx, mut prediction_shutdown_rx) = broadcast::channel(1);
    let server_shutdown_rx = shutdown_tx.subscribe();

    let prediction_handle = tokio::spawn(async move {
        loop {
            tracing::info!("Starting prediction worker...");
            tokio::select! {
                _ = prediction_worker(
                    camera_clone.clone(),
                    prediction_client.clone(),
                    config.prediction_service.get_prediction_delay_ms(),
                ) => {
                    tracing::error!("Prediction worker exited. Restarting in 5 seconds...");
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
                _ = prediction_shutdown_rx.recv() => {
                    tracing::info!("Prediction worker received shutdown signal.");
                    break;
                }
            }
        }
        tracing::info!("Prediction worker stopped.");
    });

    let server_handle = tokio::spawn(async move { server.run(server_shutdown_rx).await });

    shutdown_signal().await;
    tracing::info!("Shutdown signal received, starting graceful shutdown.");

    let _ = shutdown_tx.send(());
    let _ = prediction_handle.await;
    let _ = server_handle.await;

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
