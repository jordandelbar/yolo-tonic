use crate::{
    cv_utils::{CvUtilsError, ImageConverter},
    server::SharedState,
};
use axum::{
    body::Bytes,
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
};
use thiserror::Error;
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
    let mat = ImageConverter::bytes_to_mat(image_data.clone())
        .map_err(PredictImageError::OpenCvDecode)?;

    let predictions = state
        .prediction_service
        .predict(image_data.to_vec())
        .await
        .map_err(|e| PredictImageError::PredictionService(e.to_string()))?;

    let mut annotated_mat = mat.clone();

    ImageConverter::annotate_frame(&mut annotated_mat, &predictions)?;

    let annotated_image_data = ImageConverter::encode_mat_to_jpg(&annotated_mat)?;

    let response = Response::builder()
        .header(header::CONTENT_TYPE, "image/jpeg")
        .body(axum::body::Body::from(annotated_image_data))
        .map_err(|e| PredictImageError::HttpBuilder(e.to_string()))?;

    Ok(response)
}
