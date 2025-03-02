.PHONY: download-model local-run-server \
		local-run-client local-all local-services-up \
		services-up open-webpage all-cuda

# For local use
download-model:
	bash scripts/model_download/run.sh
	mv yolov8m.onnx yolo_prediction/models
	rm yolov8m.pt

local-run-server:
	@cd yolo_prediction && cargo run

local-run-client:
	@cd webcam_capture && cargo run

local-services-build:
	@docker compose -f compose.local.yaml build

local-services-up:
	@docker compose -f compose.local.yaml up -d

local-services-cuda-build:
	@docker compose -f compose.cuda.local.yaml build

local-services-cuda-up:
	@docker compose -f compose.cuda.local.yaml up -d

# For external use
services-up:
	@docker compose -f compose.yaml up -d

services-up-cuda:
	@docker compose -f compose.cuda.yaml up -d

open-webpage:
	@if command -v xdg-open > /dev/null; then xdg-open index.html; \
	elif command -v open > /dev/null; then open index.html; \
	elif command -v start > /dev/null; then start index.html; \
	else echo "No suitable command found to open the file."; fi

# Tear up
services-down:
	@docker compose down

all: services-up open-webpage
all-cuda: services-up-cuda open-webpage
