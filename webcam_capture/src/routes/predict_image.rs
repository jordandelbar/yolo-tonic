use crate::{
    cv_utils::{CvImage, CvUtilsError},
    server::SharedState,
};
use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
};
use thiserror::Error;
use tokio::time::Instant;
use tracing::instrument;

#[derive(Error, Debug)]
pub enum PredictImageError {
    #[error("OpenCV decode failed: {0}")]
    OpenCvDecode(CvUtilsError),
    #[error("Prediction service failed: {0}")]
    PredictionService(String),
    #[error("Image conversion failed: {0}")]
    ImageConversion(CvUtilsError),
    #[error("HTTP builder failed: {0}")]
    HttpBuilder(String),
}

impl From<CvUtilsError> for PredictImageError {
    fn from(err: CvUtilsError) -> Self {
        PredictImageError::ImageConversion(err)
    }
}

impl IntoResponse for PredictImageError {
    fn into_response(self) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Something went wrong: {}", self),
        )
            .into_response()
    }
}

#[instrument(skip(state, image_data))]
pub async fn predict_image(
    State(state): State<SharedState>,
    image_data: Bytes,
) -> Result<Response, PredictImageError> {
    let mut image =
        CvImage::from_bytes(image_data.clone()).map_err(PredictImageError::OpenCvDecode)?;

    let start = Instant::now();
    let predictions = state
        .prediction_service
        .predict(image_data.to_vec())
        .await
        .map_err(|e| PredictImageError::PredictionService(e.to_string()))?;
    let elapsed = start.elapsed().as_millis();
    state
        .metrics
        .record_prediction_duration(elapsed as u64, "predict_image");
    state.metrics.record_request("predict_image");

    let response = Response::builder()
        .header(header::CONTENT_TYPE, "image/jpeg")
        .body(Body::from(image.annotate(&predictions)?.to_jpg()?))
        .map_err(|e| PredictImageError::HttpBuilder(e.to_string()))?;

    Ok(response)
}
