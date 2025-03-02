use crate::camera::Camera;
use crate::camera::CameraError;
use bytes::Bytes;
use futures::stream;
use std::{sync::Arc, time::Duration};
use thiserror::Error;
use tokio::time::sleep;
use tracing::instrument;

const FRAME_BOUNDARY: &str = "--frame";

#[derive(Clone)]
pub struct VideoStream {
    pub camera: Arc<Camera>,
    pub video_stream_delay: u64,
}

#[derive(Error, Debug)]
pub enum VideoStreamError {
    #[error("Camera error: {0}")]
    Camera(#[from] CameraError),
    #[error("Http builder error: {0}")]
    HttpBuilderError(String),
}

impl VideoStream {
    pub fn new(camera: Arc<Camera>, video_stream_delay: u64) -> Self {
        Self {
            camera,
            video_stream_delay,
        }
    }

    #[instrument(skip(self))]
    pub fn generate_stream(self) -> impl futures::Stream<Item = Result<Bytes, VideoStreamError>> {
        let camera = self.camera.clone();

        stream::unfold(camera, move |camera| async move {
            sleep(Duration::from_millis(self.video_stream_delay)).await;
            match camera.get_annotated_frame().await {
                Ok(Some(frame)) => {
                    let part_header = format!(
                        "--{}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                        FRAME_BOUNDARY,
                        frame.len()
                    );
                    let mut body = part_header.into_bytes();
                    body.extend_from_slice(&frame);
                    body.extend_from_slice(b"\r\n");
                    Some((Ok::<_, VideoStreamError>(Bytes::from(body)), camera))
                }
                Ok(_) => None,
                Err(e) => {
                    tracing::error!("Error getting frame: {:?}", e);
                    Some((Err(VideoStreamError::from(e)), camera))
                }
            }
        })
    }
}
