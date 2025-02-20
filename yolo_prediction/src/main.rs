use yolo_prediction_service::start_app;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    start_app().await
}
