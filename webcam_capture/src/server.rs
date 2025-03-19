use crate::{
    camera::Camera,
    config::Config,
    prediction::{PredictionPoller, PredictionService},
    routes::api_routes,
    stream::VideoStream,
};
use axum::Router;
use std::sync::Arc;
use tokio::{net::TcpListener, sync::broadcast::Receiver, task::JoinHandle};

#[derive(Clone)]
pub struct SharedState {
    pub video_stream: VideoStream,
    pub prediction_service: Arc<PredictionService>,
}

pub struct HttpServer {
    router: Router,
    listener: TcpListener,
    prediction_poller: PredictionPoller,
}

impl HttpServer {
    pub async fn new(camera: Arc<Camera>, config: &Config) -> anyhow::Result<Self> {
        let addr = config.server.get_address();
        let video_stream = VideoStream::new(camera.clone(), config.server.get_stream_delay_ms());

        let prediction_service = match PredictionService::new(&config.prediction_service).await {
            Ok(service) => Arc::new(service),
            Err(e) => {
                tracing::error!("Failed to initialize prediction service: {:?}", e);
                return Err(e.into());
            }
        };

        let app_state = SharedState {
            video_stream,
            prediction_service: prediction_service.clone(),
        };

        let router = Router::new().merge(api_routes()).with_state(app_state);

        let listener = TcpListener::bind(addr).await?;

        let prediction_poller = PredictionPoller::new(
            camera.clone(),
            prediction_service,
            &config.prediction_polling,
        );

        Ok(Self {
            router,
            listener,
            prediction_poller,
        })
    }

    pub async fn run(
        self,
        shutdown_rx: Receiver<()>,
    ) -> anyhow::Result<JoinHandle<anyhow::Result<()>>> {
        tracing::info!("Starting app on {}", &self.listener.local_addr().unwrap());

        let listener = self.listener;
        let router = self.router;
        let prediction_poller = self.prediction_poller;

        let prediction_poller_handle = tokio::spawn({
            let shutdown_rx = shutdown_rx.resubscribe();
            async move {
                prediction_poller.run(shutdown_rx).await;
            }
        });

        let server_handle = tokio::spawn({
            let mut shutdown_rx = shutdown_rx.resubscribe();
            async move {
                let server = axum::serve(listener, router);
                tokio::select! {
                    result = server.with_graceful_shutdown(async move { shutdown_rx.recv().await.ok(); }) => {
                        result?;
                    }
                }
                Ok(())
            }
        });

        tokio::spawn({
            let mut shutdown_rx = shutdown_rx.resubscribe();
            async move {
                shutdown_rx.recv().await.ok();
                prediction_poller_handle.abort();
            }
        });

        Ok(server_handle)
    }
}
