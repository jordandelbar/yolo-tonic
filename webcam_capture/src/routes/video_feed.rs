use crate::{camera::Camera, server::SharedState};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
};
use futures::TryStreamExt;

pub async fn video_feed(
    ws: WebSocketUpgrade,
    State(state): State<SharedState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: SharedState) {
    let camera = match Camera::new(0, state.prediction_service.clone(), &state.camera_config) {
        Ok(cam) => cam,
        Err(e) => {
            tracing::error!("Could not create camera: {:?}", e);
            return;
        }
    };

    let (frame_thread, prediction_thread) = match camera.start().await {
        Ok(threads) => threads,
        Err(e) => {
            tracing::error!("Camera start error: {:?}", e);
            return;
        }
    };

    let mut stream = camera.subscribe();

    while let Some(Ok(frame)) = stream.try_next().await.transpose() {
        if socket.send(Message::Binary(frame.into())).await.is_err() {
            tracing::error!("Failed to send frame to client");
            break;
        }
    }

    camera.stop();
    let _ = frame_thread.await;
    let _ = prediction_thread.await;
    tracing::info!("Client disconnected");
}
