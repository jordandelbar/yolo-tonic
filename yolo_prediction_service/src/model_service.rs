use crate::proto::{ImageFrame, PredictionBatch};
use tonic::{async_trait, Status};

#[async_trait]
pub trait ModelService: Send + Sync + Clone + 'static {
    async fn predict(&self, frame: ImageFrame) -> Result<PredictionBatch, Status>;
}
