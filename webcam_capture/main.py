import time
import cv2
import grpc
import threading

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from loguru import logger

import yolo_service_pb2 as yolo_service
import yolo_service_pb2_grpc as yolo_service_grpc

app = FastAPI()

last_prediction = None
last_prediction_lock = threading.Lock()
latest_frame = None
frame_lock = threading.Lock()

# TODO: environment variable
server_address = "localhost:50051"
channel = grpc.insecure_channel(server_address)
stub = yolo_service_grpc.YoloServiceStub(channel)

cap = cv2.VideoCapture(0)

def send_prediction_request(stop_prediction_request):
    global last_prediction
    logger.info("prediction thread started")
    while not stop_prediction_request.is_set():
        time.sleep(0.1)

        with frame_lock:
            if latest_frame is None:
                continue
            frame_copy = latest_frame.copy()

        try:
            _, encoded_image = cv2.imencode(".jpg", frame_copy)
            image_bytes = encoded_image.tobytes()
            image_frame = yolo_service.ImageFrame(
                image_data=image_bytes, timestamp=int(time.time() * 1000)
            )

            prediction = stub.Predict(image_frame)

            with last_prediction_lock:
                last_prediction = prediction.detections

        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e}")
        except Exception as e:
            logger.error(f"prediction request error: {e}")


@app.get("/video_feed")
async def video_feed(request: Request):
    logger.info("client connected. Starting stream")
    return StreamingResponse(generate_frames(request), media_type="multipart/x-mixed-replace; boundary=frame")

async def generate_frames(request: Request):
    global latest_frame, last_prediction, cap
    while True:
        stop_prediction_request = threading.Event()
        prediction_thread = threading.Thread(target=send_prediction_request, args=(stop_prediction_request,), daemon=True)
        prediction_thread.start()

        try:
            while True:
                if cap is None:
                    time.sleep(0.1)
                    continue

                if await request.is_disconnected():
                    logger.info("client disconnected. Stopping stream")
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning("failed to capture frame, skipping")
                    time.sleep(0.1)
                    continue

                with frame_lock:
                    latest_frame = frame.copy()

                with last_prediction_lock:
                    current_prediction = last_prediction

                if current_prediction:
                    for detection in current_prediction:
                        x1, y1, x2, y2 = map(int, [detection.x1, detection.y1, detection.x2, detection.y2])
                        class_label, confidence = detection.class_label, detection.confidence

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, str(class_label), (x1, max(20, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"{confidence:.2f}", (x1, min(frame.shape[0] - 10, y2 + 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                ret, encoded_frame = cv2.imencode(".jpg", frame)
                frame_bytes = encoded_frame.tobytes()
                yield (b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                time.sleep(0.033)
        except Exception as e:
            logger.error(f"error in generate frame: {e}")
        finally:
            stop_prediction_request.set()
            prediction_thread.join()
            logger.info("prediction thread stopped")
        if await request.is_disconnected():
            break
            logger.info("restarting prediction thread for new stream")

@app.on_event("startup")
async def startup_event():
    global cap
    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        logger.info("waiting for camera to initialize")
        time.sleep(0.5)
    else:
        logger.error("unable to initialize camera.")
        exit(1)


@app.on_event("shutdown")
async def shutdown_event():
    global cap
    cap.release()
    channel.close()
