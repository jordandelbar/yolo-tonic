import time

import cv2
import grpc
import yolo_service_pb2 as yolo_service
import yolo_service_pb2_grpc as yolo_service_grpc

def predict_stream(stub):
    cap = cv2.VideoCapture(0)

    def frame_generator():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _, encoded_image = cv2.imencode(".jpg", frame)
            image_bytes = encoded_image.tobytes()

            image_frame = yolo_service.ImageFrame(image_data=image_bytes, timestamp=int(time.time_ns() / 1e9))

            yield image_frame

    predictions = stub.PredictStream(frame_generator())
    for batch in predictions:
        ret, frame = cap.read()
        for detection in batch.detections:
            x1 = int(detection.x1)
            y1 = int(detection.y1)
            x2 = int(detection.x2)
            y2 = int(detection.y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(detection.class_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("YOLO Predictions", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    server_address = "localhost:50051"
    channel = grpc.insecure_channel(server_address)
    stub = yolo_service_grpc.YoloServiceStub(channel)
    predict_stream(stub)
