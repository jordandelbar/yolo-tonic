.PHONY: run-server run-client

run-server:
	@cd server && cargo run

run-client:
	@cd client && uv run python main.py
