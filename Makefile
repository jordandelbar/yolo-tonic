.PHONY: run-server run-client

download-model:
	bash scripts/model_download.sh

run-server:
	@cd server && cargo run

run-client:
	@cd client && uv run python main.py
