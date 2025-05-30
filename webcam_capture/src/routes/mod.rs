use crate::server::SharedState;
mod health;
mod metrics;
mod predict_image;
mod video_feed;

use health::healthcheck;
use metrics::metrics_handler;
use predict_image::predict_image;
use video_feed::video_feed;

use axum::{
    routing::{get, post},
    Router,
};

pub fn api_routes() -> Router<SharedState> {
    Router::new()
        .route("/ws/video_feed", get(video_feed))
        .route("/predict_image", post(predict_image))
        .route("/health", get(healthcheck))
        .route("/metrics", get(metrics_handler))
}
