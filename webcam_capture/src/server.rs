use crate::{camera::Camera, config::ServerConfig, routes::api_routes, stream::VideoStream};
use axum::Router;
use std::sync::Arc;
use tokio::{net::TcpListener, sync::broadcast::Receiver, task::JoinHandle};

pub struct HttpServer {
    router: Router,
    listener: TcpListener,
}

impl HttpServer {
    pub async fn new(camera: Arc<Camera>, server_config: &ServerConfig) -> anyhow::Result<Self> {
        let addr = server_config.get_address();
        let video_stream = VideoStream::new(camera, server_config.get_stream_delay_ms());
        let router = Router::new().merge(api_routes()).with_state(video_stream);

        let listener = TcpListener::bind(addr).await?;

        Ok(Self { router, listener })
    }

    pub async fn run(
        self,
        mut shutdown_rx: Receiver<()>,
    ) -> anyhow::Result<JoinHandle<anyhow::Result<()>>> {
        tracing::info!("Starting app on {}", &self.listener.local_addr().unwrap());

        let listener = self.listener;
        let router = self.router;

        let handle = tokio::spawn(async move {
            let server = axum::serve(listener, router);

            tokio::select! {
                result = server.with_graceful_shutdown(async move {shutdown_rx.recv().await.ok();}) => {
                    result?;
                }
            }

            Ok(())
        });

        Ok(handle)
    }
}
