use crate::{
    config::get_configuration, inference_service::InferenceService, model_service::ModelService,
    ort_service::OrtModelService, proto::yolo_service_server::YoloServiceServer,
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
    let config = get_configuration().expect("failed to load config");
    let ort_model_service =
        OrtModelService::new(&config).expect("failed to instantiate ort model service");

    let addr = config.service.get_address();
    let mut app = App::new(ort_model_service, &addr);
    tracing::info!("listening on {}", &addr);

    app.run().await?;

    Ok(())
}
