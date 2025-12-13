FROM nvcr.io/nvidia/tensorrt:25.03-py3 AS builder

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    protobuf-compiler \
    libprotobuf-dev \
    cmake \
    git \
    wget \
    pkg-config \
    libssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN rustup toolchain install 1.85.0 --profile minimal && \
    rustup default 1.85.0

WORKDIR /app

RUN curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-gpu-1.23.2.tgz" \
    && mkdir /app/onnxruntime \
    && tar -xzf onnxruntime.tgz --strip-components=1 -C /app/onnxruntime \
    && rm onnxruntime.tgz

COPY ./yolo_prediction /app/yolo_prediction
COPY ./yolo_proto /app/yolo_proto
WORKDIR /app/yolo_prediction

ENV ORT_LIB_LOCATION=/app/onnxruntime
RUN cargo build --release && strip target/release/yolo_prediction
RUN rm -rf /root/.cargo/registry /root/.cargo/git /app/yolo_prediction/target/debug

RUN GRPC_HEALTH_PROBE_VERSION=v0.4.13 && \
    wget -qO/bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 && \
    chmod +x /bin/grpc_health_probe

FROM nvcr.io/nvidia/tensorrt:25.03-py3

WORKDIR /app

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends protobuf-compiler libprotobuf-dev libgomp1 \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/onnxruntime/lib/* /usr/lib/

COPY --from=builder /app/yolo_prediction/target/release/yolo_prediction /app/yolo_prediction
COPY --from=builder /app/yolo_prediction/configuration/ /app/configuration
COPY --from=builder /bin/grpc_health_probe /bin/grpc_health_probe
COPY ./yolo_prediction/models/yolov8m.onnx /app/models/yolov8m.onnx
COPY ./yolo_prediction/labels/yolov8_labels.txt /app/labels/yolov8_labels.txt
COPY ./yolo_prediction/healthcheck.sh /app/healthcheck.sh
RUN chmod +x /app/healthcheck.sh

EXPOSE 50051

HEALTHCHECK --interval=10s --timeout=5s --retries=60 CMD /app/healthcheck.sh

CMD ["./yolo_prediction"]
