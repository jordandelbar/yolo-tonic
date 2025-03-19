use crate::{server::SharedState, stream::VideoStreamError};
use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
};
use tracing::instrument;

const CONTENT_TYPE: &str = "multipart/x-mixed-replace; boundary=frame";

#[instrument(skip(state))]
pub async fn video_feed(State(state): State<SharedState>) -> Result<Response, VideoStreamError> {
    let stream = state.video_stream.generate_stream();

    let body = Body::from_stream(stream);

    let response = Response::builder()
        .header(header::CONTENT_TYPE, CONTENT_TYPE)
        .body(body)
        .map_err(|e| VideoStreamError::HttpBuilderError(e.to_string()))?;

    Ok(response)
}

impl IntoResponse for VideoStreamError {
    fn into_response(self) -> Response {
        let status = match self {
            VideoStreamError::Camera(_) => StatusCode::INTERNAL_SERVER_ERROR,
            VideoStreamError::HttpBuilderError(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, self.to_string()).into_response()
    }
}
