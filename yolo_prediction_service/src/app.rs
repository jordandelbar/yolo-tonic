use crate::{InferenceService, ModelService, YoloServiceServer};
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
        let addr = self.addr.parse().unwrap();

        println!("Inference service listening on {}", self.addr);

        Server::builder()
            .add_service(YoloServiceServer::new(self.inference_service.clone()))
            .serve(addr)
            .await?;

        Ok(())
    }
}
