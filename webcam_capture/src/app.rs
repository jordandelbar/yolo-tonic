use crate::{camera::Camera, config::Config, server::HttpServer};

use std::{error::Error, sync::Arc};
use tokio::{signal, sync::broadcast};

pub async fn start_app(config: Config) -> Result<(), Box<dyn Error>> {
    let camera: Arc<Camera> = match Camera::new().await {
        Ok(cam) => Arc::new(cam),
        Err(e) => {
            tracing::error!("Failed to initialize camera: {:?}", e);
            return Err(Box::new(e));
        }
    };

    let server = HttpServer::new(camera.clone(), &config).await?;

    let (shutdown_tx, _) = broadcast::channel(1);
    let server_shutdown_rx = shutdown_tx.subscribe();

    let server_handle = server.run(server_shutdown_rx).await?;

    shutdown_signal().await;
    tracing::info!("Shutdown signal received, starting graceful shutdown.");

    let _ = shutdown_tx.send(());
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
