use crate::stream::VideoStream;
use axum::{
    body::Body,
    extract::State,
    response::{IntoResponse, Response},
};
use futures::stream::TryStreamExt;
use tracing::instrument;

#[instrument(skip(video_stream))]
pub async fn video_feed(State(video_stream): State<VideoStream>) -> impl IntoResponse {
    let content_type = "multipart/x-mixed-replace; boundary=frame";

    let stream = video_stream.generate_stream();

    let stream = stream.map_err(|e| {
        tracing::error!("Stream error: {:?}", e);
        anyhow::Error::msg(e.to_string())
    });

    let body = Body::from_stream(stream);

    let response = Response::builder()
        .header(axum::http::header::CONTENT_TYPE, content_type)
        .body(body)
        .unwrap();

    response
}
