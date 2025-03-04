use axum::{response::IntoResponse, response::Json};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Status {
    status: String,
}

pub async fn healthcheck() -> impl IntoResponse {
    Json(Status {
        status: "Available".into(),
    })
}
