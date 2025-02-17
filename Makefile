.PHONY: run-server run-client

download-model:
	bash scripts/model_download/run.sh
	mv yolov8m.onnx yolo_prediction_service/models
	rm yolov8m.pt

run-server:
	@cd yolo_prediction_service && cargo run

run-client:
	@cd webcam_capture && uv run python main.py
