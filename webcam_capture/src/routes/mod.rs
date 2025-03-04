use crate::stream::VideoStream;
mod health;
mod video_feed;

use health::healthcheck;
use video_feed::video_feed;

use axum::{routing::get, Router};

pub fn api_routes() -> Router<VideoStream> {
    Router::new()
        .route("/video_feed", get(video_feed))
        .route("/health", get(healthcheck))
}
