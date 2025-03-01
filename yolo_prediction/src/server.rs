use crate::{config::Settings, inference_service::InferenceService, ort_service::OrtModelService};
use tokio::signal;
use tonic::transport::server::Router;
use tonic::transport::Server;
use yolo_proto::yolo_service_server::YoloServiceServer;

pub struct GrpcServer {
    router: Router,
    addr: String,
}

impl GrpcServer {
    pub fn new(model_service: OrtModelService, addr: &str) -> Self {
        let inference_service = InferenceService::new(model_service);
        let router = Server::builder().add_service(YoloServiceServer::new(inference_service));

        Self {
            router,
            addr: addr.to_string(),
        }
    }

    pub async fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        let addr = self.addr.parse().expect("failed to parse address");

        tracing::info!("Inference service listening on {}", self.addr);

        let shutdown = async {
            shutdown_signal().await;
            tracing::info!("Shutdown signal received, starting graceful shutdown")
        };

        self.router.serve_with_shutdown(addr, shutdown).await?;
        Ok(())
    }
}

pub async fn start_server(config: Settings) -> Result<(), Box<dyn std::error::Error>> {
    let ort_model_service =
        OrtModelService::new(&config).expect("failed to instantiate ort model service");

    let addr = config.service.get_address();
    let grpc_server = GrpcServer::new(ort_model_service, &addr);
    tracing::info!("Listening on {}", &addr);

    grpc_server.run().await?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
