.PHONY: run-server run-client

download-model:
	bash scripts/model_download.sh

run-server:
	@cd yolo_prediction_service && cargo run

run-client:
	@cd webcam_capture && uv run python main.py
