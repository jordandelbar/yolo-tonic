use crate::{model_service::ModelService, state::State};
use std::sync::Arc;
use tonic::{async_trait, Request, Response, Status};
use yolo_proto::{
    yolo_service_server::YoloService, Empty, ImageFrame, PredictionBatch, YoloClassLabels,
};

#[derive(Debug, Clone)]
pub struct InferenceService<M: ModelService, S: State> {
    model_service: Arc<M>,
    service_state: Arc<S>,
}

impl<M: ModelService, S: State> InferenceService<M, S> {
    pub fn new(model_service: M, state: S) -> Result<Self, String> {
        Ok(Self {
            model_service: Arc::new(model_service),
            service_state: Arc::new(state),
        })
    }
}

#[async_trait]
impl<M: ModelService, S: State> YoloService for InferenceService<M, S> {
    async fn predict(
        &self,
        request: Request<ImageFrame>,
    ) -> Result<Response<PredictionBatch>, Status> {
        let image_frame = request.into_inner();
        let model_service = self.model_service.clone();
        let batch = model_service.predict(image_frame).await?;

        tracing::debug!("Returning {} detections", batch.detections.len());
        for (i, detection) in batch.detections.iter().enumerate() {
            tracing::debug!(
                "Detection {}: class_id={}, confidence={:.3}, bbox=({:.1}, {:.1}, {:.1}, {:.1})",
                i,
                detection.class_id,
                detection.confidence,
                detection.x1,
                detection.y1,
                detection.x2,
                detection.y2
            );
        }

        Ok(Response::new(batch))
    }

    async fn get_yolo_class_labels(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<YoloClassLabels>, Status> {
        let labels = self.service_state.get_labels().clone();
        let response = YoloClassLabels {
            class_labels: labels,
        };

        Ok(Response::new(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LabelsConfig;
    use std::path::PathBuf;

    use yolo_proto::{BoundingBox, ColorLabel};

    #[derive(Clone)]
    struct MockModelService {}

    #[async_trait]
    impl ModelService for MockModelService {
        async fn predict(&self, frame: ImageFrame) -> Result<PredictionBatch, Status> {
            let detections = vec![
                BoundingBox {
                    class_id: 7,
                    confidence: 0.95,
                    x1: 10.0,
                    y1: 20.0,
                    x2: 100.0,
                    y2: 150.0,
                },
                BoundingBox {
                    class_id: 42,
                    confidence: 0.88,
                    x1: 200.0,
                    y1: 50.0,
                    x2: 300.0,
                    y2: 200.0,
                },
            ];

            Ok(PredictionBatch {
                detections,
                timestamp: frame.timestamp,
            })
        }
    }

    pub struct MockState {
        class_labels: Vec<ColorLabel>,
    }

    impl State for MockState {
        fn new(_labels_file: &LabelsConfig) -> Result<Self, String> {
            Ok(MockState {
                class_labels: vec![
                    ColorLabel {
                        label: "class1".to_string(),
                        red: 255,
                        green: 0,
                        blue: 0,
                    },
                    ColorLabel {
                        label: "class2".to_string(),
                        red: 255,
                        green: 0,
                        blue: 0,
                    },
                    ColorLabel {
                        label: "class3".to_string(),
                        red: 255,
                        green: 0,
                        blue: 0,
                    },
                ],
            })
        }

        fn get_labels(&self) -> &Vec<ColorLabel> {
            &self.class_labels
        }
    }

    #[tokio::test]
    async fn test_predict() -> Result<(), Box<dyn std::error::Error>> {
        let mock_labels_config = LabelsConfig {
            labels_file: "dummy_labels.txt".to_string(),
            labels_dir: PathBuf::from("./dummy_labels_dir"),
        };

        let mock_model = MockModelService {};
        let mock_state = MockState::new(&mock_labels_config).unwrap();
        let inference_service = InferenceService::new(mock_model, mock_state)?;

        let image_frame = ImageFrame {
            image_data: vec![0; 100],
            timestamp: 12345,
        };

        let request = Request::new(image_frame);
        let response = inference_service.predict(request).await?;

        let batch = response.into_inner();
        assert_eq!(batch.detections.len(), 2);
        assert_eq!(batch.detections[0].class_id, 7);
        assert_eq!(batch.detections[1].class_id, 42);

        Ok(())
    }
}
