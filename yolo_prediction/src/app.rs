use crate::{
    inference_service::InferenceService, model_service::ModelService, ort_service::OrtModelService,
    proto::yolo_service_server::YoloServiceServer,
};
use tonic::transport::Server;

pub struct App<M: ModelService> {
    inference_service: InferenceService<M>,
    addr: String,
}

impl<M: ModelService> App<M> {
    pub fn new(model_service: M, addr: &str) -> Self {
        let inference_service = InferenceService::new(model_service);
        Self {
            inference_service,
            addr: addr.to_string(),
        }
    }

    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let addr = self.addr.parse().expect("failed to parse address");

        println!("Inference service listening on {}", self.addr);

        Server::builder()
            .add_service(YoloServiceServer::new(self.inference_service.clone()))
            .serve(addr)
            .await?;

        Ok(())
    }
}

pub async fn start_app() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "./models/yolov8m.onnx";
    let ort_model_service =
        OrtModelService::new(model_path, 5).expect(&format!("no model found at {}", model_path));

    // TODO: config for local dev vs docker
    // let addr = "[::1]:50051";
    let addr = "0.0.0.0:50051";
    let mut app = App::new(ort_model_service, addr);
    tracing::info!("listening on {}", addr);

    app.run().await?;

    Ok(())
}
