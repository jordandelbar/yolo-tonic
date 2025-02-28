use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use webcam_capture::{config, start_app};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = config::get_configuration().expect("failed to load config");
    let log_level = &format!("{},ort=info", config.log_level.as_str());

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| log_level.into()),
        )
        .with(tracing_subscriber::fmt::layer().json().with_level(true))
        .init();

    start_app(config).await?;

    Ok(())
}
