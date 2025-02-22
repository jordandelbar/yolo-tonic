use tonic::{async_trait, Status};
use yolo_proto::{ImageFrame, PredictionBatch};

#[async_trait]
pub trait ModelService: Send + Sync + Clone + 'static {
    async fn predict(&self, frame: ImageFrame) -> Result<PredictionBatch, Status>;
}
