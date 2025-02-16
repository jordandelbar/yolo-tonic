use yolo_prediction_service::{App, OrtModelService};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ort_model_service = OrtModelService::new("./models/yolov8m.onnx", 5).unwrap();

    let mut app = App::new(ort_model_service, "[::1]:50051");
    app.run().await?;

    Ok(())
}
