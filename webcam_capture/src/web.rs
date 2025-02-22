use crate::camera::Camera;
use crate::config::Settings;
use crate::prediction::prediction_worker;
use crate::routes::video_stream;
use axum::routing::{get, Router};
use std::error::Error;
use std::sync::Arc;
use tokio::net::TcpListener;

pub async fn start_app(config: Settings) -> Result<(), Box<dyn Error>> {
    let camera = Arc::new(Camera::new().await);

    let camera_clone = camera.clone();
    tokio::spawn(async move {
        prediction_worker(camera_clone, config.prediction_service.clone()).await;
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
