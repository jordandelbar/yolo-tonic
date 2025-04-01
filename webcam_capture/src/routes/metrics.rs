use crate::server::SharedState;
use axum::{extract::State, response::IntoResponse};
use prometheus::{Encoder, TextEncoder};

pub async fn metrics_handler(State(state): State<SharedState>) -> impl IntoResponse {
    let metric_families = state.metrics.registry.gather();

    let mut buffer = Vec::new();
    let encoder = TextEncoder::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();

    String::from_utf8(buffer).unwrap().into_response()
}
