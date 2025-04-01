use crate::{
    config::{CameraConfig, Config},
    prediction::PredictionService,
    routes::api_routes,
    telemetry::Metrics,
};
use axum::Router;
use axum_otel_metrics::HttpMetricsLayerBuilder;
use std::sync::Arc;
use tokio::{net::TcpListener, sync::broadcast::Receiver, task::JoinHandle};

#[derive(Clone)]
pub struct SharedState {
    pub prediction_service: Arc<PredictionService>,
    pub camera_config: CameraConfig,
    pub metrics: Arc<Metrics>,
}

pub struct HttpServer {
    router: Router,
    listener: TcpListener,
}

impl HttpServer {
    pub async fn new(
        prediction_service: Arc<PredictionService>,
        config: &Config,
    ) -> anyhow::Result<Self> {
        let addr = config.server.get_address();

        let metrics = Arc::new(Metrics::new());
        let metrics_layer = HttpMetricsLayerBuilder::new().build();

        let app_state = SharedState {
            prediction_service,
            camera_config: config.camera.clone(),
            metrics,
        };

        let router = Router::new()
            .merge(api_routes())
            .with_state(app_state)
            .layer(metrics_layer);

        let listener = TcpListener::bind(addr).await?;

        Ok(Self { router, listener })
    }

    pub async fn run(
        self,
        shutdown_rx: Receiver<()>,
    ) -> anyhow::Result<JoinHandle<anyhow::Result<()>>> {
        tracing::info!("Starting app on {}", &self.listener.local_addr().unwrap());

        let listener = self.listener;
        let router = self.router;
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
            }
        });

        Ok(server_handle)
    }
}
