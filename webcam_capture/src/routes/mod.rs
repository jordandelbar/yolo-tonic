use crate::camera::Camera;
use crate::camera::CameraError;
use axum::{
    body::Body,
    extract::State,
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures::stream;
use futures::stream::TryStreamExt;
use std::{sync::Arc, time::Duration};
use tokio::time::sleep;
use tracing::instrument;

#[instrument(skip(camera))]
pub async fn video_stream(State(camera): State<Arc<Camera>>) -> impl IntoResponse {
    let boundary = "--frame";
    let content_type = "multipart/x-mixed-replace; boundary=frame";

    let stream = stream::unfold(camera, move |camera| async move {
        sleep(Duration::from_millis(20)).await;
        match camera.get_annotated_frame().await {
            Ok(Some(frame)) => {
                let part_header = format!(
                    "--{}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                    boundary,
                    frame.len()
                );
                let mut body = part_header.into_bytes();
                body.extend_from_slice(&frame);

                Some((Ok::<_, CameraError>(Bytes::from(body)), camera))
            }
            Ok(None) => None,
            Err(e) => {
                tracing::error!("Error getting frame: {:?}", e);
                Some((Err(e), camera))
            }
        }
    });

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
