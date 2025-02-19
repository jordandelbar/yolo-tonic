generate-prot:
	uv run python -m grpc_tools.protoc -I../yolo_prediction/proto --python_out=./src --grpc_python_out=./src ../yolo_prediction/proto/yolo_service.proto

generate-requirements:
	uv export --format requirements-txt --frozen -o requirements.txt
