use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use yolo_prediction_service::start_app;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ort=info".into()),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .json()
                .with_target(false)
                .with_level(true)
                .with_thread_names(true)
                .with_thread_ids(true),
        )
        .init();

    start_app().await
}
