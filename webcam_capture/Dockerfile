FROM rust:1.83.0 AS builder

WORKDIR /app

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends protobuf-compiler libprotobuf-dev libopencv-dev llvm clang libclang-dev

COPY ./webcam_capture /app/webcam_capture
COPY ./yolo_proto /app/yolo_proto

WORKDIR /app/webcam_capture
RUN cargo build --release

FROM debian:bookworm-slim

WORKDIR /app

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends protobuf-compiler libprotobuf-dev libgomp1 libopencv-dev llvm clang libclang-dev \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/webcam_capture/target/release/webcam_capture /app/webcam_capture
COPY --from=builder /app/webcam_capture/configuration/ /app/configuration

EXPOSE 50051

CMD ["./webcam_capture"]
