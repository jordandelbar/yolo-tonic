.PHONY: download-model local-run-server \
		local-run-client local-services-build \
		local-services-up local-services-cuda-build \
		local-services-cuda-up \
		services-up services-cuda-up \
	 	open-webpage open-grafana predict-image \
		grpc-ui services-down services-cuda-down \
		local-services-down local-services-cuda-down \
		all cuda-all

# For local use
download-model:
	cd scripts/model_download && \
		uv sync --frozen && \
		uv run main.py

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

services-cuda-up:
	@docker compose -f compose.cuda.yaml up -d

open-webpage:
	@if command -v xdg-open > /dev/null; then xdg-open index.html; \
	elif command -v open > /dev/null; then open index.html; \
	elif command -v start > /dev/null; then start index.html; \
	else echo "No suitable command found to open the file."; fi

open-grafana:
	@if command -v xdg-open > /dev/null; then xdg-open http://localhost:3000/dashboards; \
	elif command -v open > /dev/null; then open http://localhost:3000/dashboards; \
	elif command -v start > /dev/null; then start http://localhost:3000/dashboards; \
	else echo "No suitable command found to open the file."; fi

predict-image:
	@cd scripts/predict_image && bash predict_image.sh

grpc-ui:
	@cd scripts/grpc_ui && bash run_grpc_ui.sh

# Tear up
services-down:
	@docker compose -f compose.yaml down

services-cuda-down:
	@docker compose -f compose.yaml down

local-services-down:
	@docker compose -f compose.local.yaml down

local-services-cuda-down:
	@docker compose -f compose.cuda.local.yaml down

all: services-up open-webpage
all-cuda: services-cuda-up open-webpage
