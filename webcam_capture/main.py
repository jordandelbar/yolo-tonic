import time
import cv2
import grpc
import threading

from loguru import logger

import yolo_service_pb2 as yolo_service
import yolo_service_pb2_grpc as yolo_service_grpc

last_prediction = None
last_prediction_lock = threading.Lock()
latest_frame = None
frame_lock = threading.Lock()

# TODO: environment variable
server_address = "localhost:50051"
channel = grpc.insecure_channel(server_address)
stub = yolo_service_grpc.YoloServiceStub(channel)


def send_prediction_request():
    global last_prediction
    while True:
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
            logger.error(f"Prediction request error: {e}")


def display_thread():
    global latest_frame
    cap = cv2.VideoCapture(0)

    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        logger.info("waiting for camera to initialize")
        time.sleep(0.5)
    else:
        logger.error("unable to initialize camera.")
        return

    threading.Thread(target=send_prediction_request, daemon=True).start()

    while True:
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
                x1, y1, x2, y2 = map(
                    int, [detection.x1, detection.y1, detection.x2, detection.y2]
                )
                class_label, confidence = detection.class_label, detection.confidence

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    str(class_label),
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"{confidence:.2f}",
                    (x1, min(frame.shape[0] - 10, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        logger.info(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    display_thread()
