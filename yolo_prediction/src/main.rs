use yolo_prediction::start_service;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    start_service().await
}
