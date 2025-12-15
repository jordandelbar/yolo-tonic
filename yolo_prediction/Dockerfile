FROM rust:1.92.0 AS builder

WORKDIR /app

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends protobuf-compiler libprotobuf-dev

# Download and extract ONNX Runtime
RUN curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-gpu-1.23.2.tgz" \
    && mkdir /app/onnxruntime \
    && tar -xzf onnxruntime.tgz --strip-components=1 -C /app/onnxruntime \
    && rm onnxruntime.tgz

COPY ./yolo_prediction /app/yolo_prediction
COPY ./yolo_proto /app/yolo_proto

WORKDIR /app/yolo_prediction
RUN cargo build --release

FROM debian:bookworm-slim

WORKDIR /app

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends protobuf-compiler libprotobuf-dev libgomp1 \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/onnxruntime/lib/* /usr/lib/

COPY --from=builder /app/yolo_prediction/target/release/yolo_prediction /app/yolo_prediction
COPY --from=builder /app/yolo_prediction/configuration/ /app/configuration
COPY ./yolo_prediction/models/yolov8m.onnx /app/models/yolov8m.onnx
COPY ./yolo_prediction/labels/yolov8_labels.txt /app/labels/yolov8_labels.txt

EXPOSE 50051

CMD ["./yolo_prediction"]
