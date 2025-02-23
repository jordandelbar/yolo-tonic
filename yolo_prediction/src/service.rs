use crate::{
    config::Settings, inference_service::InferenceService, model_service::ModelService,
    ort_service::OrtModelService,
};
use tonic::transport::Server;
use yolo_proto::yolo_service_server::YoloServiceServer;

pub struct Service<M: ModelService> {
    inference_service: InferenceService<M>,
    addr: String,
}

impl<M: ModelService> Service<M> {
    pub fn new(model_service: M, addr: &str) -> Self {
        let inference_service = InferenceService::new(model_service);
        Self {
            inference_service,
            addr: addr.to_string(),
        }
    }

    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let addr = self.addr.parse().expect("failed to parse address");

        tracing::info!("Inference service listening on {}", self.addr);

        Server::builder()
            .add_service(YoloServiceServer::new(self.inference_service.clone()))
            .serve(addr)
            .await?;

        Ok(())
    }
}

pub async fn start_service(config: Settings) -> Result<(), Box<dyn std::error::Error>> {
    let ort_model_service =
        OrtModelService::new(&config).expect("failed to instantiate ort model service");

    let addr = config.service.get_address();
    let mut app = Service::new(ort_model_service, &addr);
    tracing::info!("listening on {}", &addr);

    app.run().await?;

    Ok(())
}
