use yolo_service::{App, MockModelService};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ort_model_service = MockModelService {};

    let mut app = App::new(ort_model_service, "[::1]:50051");
    app.run().await?;

    Ok(())
}
