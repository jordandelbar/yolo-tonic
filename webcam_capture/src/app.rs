use crate::camera::Camera;
use crate::config::Config;
use crate::prediction::PredictionService;
use crate::server::HttpServer;

use std::{error::Error, sync::Arc};
use tokio::{signal, sync::broadcast};

pub async fn start_app(config: Config) -> Result<(), Box<dyn Error>> {
    let camera = match Camera::new().await {
        Ok(cam) => Arc::new(cam),
        Err(e) => {
            tracing::error!("Failed to initialize camera: {:?}", e);
            return Err(Box::new(e));
        }
    };

    let server = HttpServer::new(camera.clone(), config.clone()).await?;

    let (shutdown_tx, mut prediction_shutdown_rx) = broadcast::channel(1);
    let server_shutdown_rx = shutdown_tx.subscribe();

    let mut prediction_service = PredictionService::new(
        camera.clone(),
        config.prediction_service.get_address(),
        config.prediction_service.get_prediction_delay_ms(),
    )
    .await?;

    let prediction_handle = tokio::spawn(async move {
        tokio::select! {
            _ = prediction_service.run() => {},
            _ = prediction_shutdown_rx.recv() => {
                tracing::info!("Prediction worker received shutdown signal.");
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
