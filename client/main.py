import time

import cv2
import grpc
import yolo_service_pb2 as yolo_service
import yolo_service_pb2_grpc as yolo_service_grpc

from loguru import logger

def predict_stream(stub):
    cap = cv2.VideoCapture(0)

    retries = 10
    while retries > 0:
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        logger.info("waiting for camera to initialize")
        time.sleep(0.5)
        retries -= 1
    else:
        logger.error("unable to initialize camera.")
        return

    def frame_generator():
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("failed to capture frame, skipping")
                time.sleep(0.1)
                continue

            _, encoded_image = cv2.imencode(".jpg", frame)
            image_bytes = encoded_image.tobytes()

            image_frame = yolo_service.ImageFrame(image_data=image_bytes, timestamp=int(time.time_ns() / 1e9))

            yield image_frame

    try:
        predictions = stub.PredictStream(frame_generator())
        for batch in predictions:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("failed to capture frame, skipping this batch")
                continue

            for detection in batch.detections:
                x1 = int(detection.x1)
                y1 = int(detection.y1)
                x2 = int(detection.x2)
                y2 = int(detection.y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    str(detection.class_label),
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    lineType=cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    f"{detection.confidence:.2f}",
                    (x1, min(frame.shape[0] - 10, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    lineType=cv2.LINE_AA
                )

            cv2.imshow("YOLO Predictions", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    server_address = "localhost:50051"
    channel = grpc.insecure_channel(server_address)
    stub = yolo_service_grpc.YoloServiceStub(channel)
    predict_stream(stub)
