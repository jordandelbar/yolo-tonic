import asyncio
import cv2
import logging

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from src.camera import Camera
from src.prediction import PredictionManager
from src.custom_logging import CustomizeLogger
from src.config import cfg


logger = logging.getLogger(__name__)

config_path = Path(__file__).with_name("logging_config.json")
logger = CustomizeLogger.make_logger(config_path, cfg.environment)

camera = Camera()
prediction_manager = PredictionManager(camera)

active_clients = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await camera.initialize()
    yield
    await camera.release()


app = FastAPI(title="Yolo webcam capture", lifespan=lifespan, debug=False)


@app.get("/video_feed")
async def video_feed(request: Request):
    client_id = id(request)
    logger.info(f"client {client_id} connected, starting stream")

    if not active_clients:
        await prediction_manager.start()
    active_clients.add(client_id)

    try:
        return StreamingResponse(
            generate_frames(request, client_id),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
    except Exception as e:
        logger.error(f"error in video feed: {e}")
        active_clients.discard(client_id)
        if not active_clients:
            await prediction_manager.stop()
        raise


async def generate_frames(request: Request, client_id: int):
    try:
        async for frame in camera.frames():
            if await request.is_disconnected():
                logger.info("client disconnected, stopping stream")
                break

            prediction = await prediction_manager.predict(frame)
            if prediction:
                draw_detection_boxes(frame, prediction)

            ret, encoded_frame = cv2.imencode(".jpg", frame)
            frame_bytes = encoded_frame.tobytes()
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            await asyncio.sleep(0.033)
    finally:
        active_clients.discard(client_id)
        if not active_clients:
            await prediction_manager.stop()
        logger.info(f"client {client_id} stream ended")


def draw_detection_boxes(frame, prediction):
    for detection in prediction:
        x1, y1, x2, y2 = map(
            int,
            [detection.x1, detection.y1, detection.x2, detection.y2],
        )
        class_label, confidence = (
            detection.class_label,
            detection.confidence,
        )
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
