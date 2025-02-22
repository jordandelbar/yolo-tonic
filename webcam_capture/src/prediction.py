import logging
import time
import asyncio
import grpc
import cv2

import src.yolo_service_pb2 as yolo_service
import src.yolo_service_pb2_grpc as yolo_service_grpc

from src.config import cfg

logger = logging.getLogger(__name__)


class PredictionManager:
    def __init__(self, camera):
        self.camera = camera
        self.channel = None
        self.stub = None
        self.last_prediction = None
        self.last_prediction_lock = asyncio.Lock()
        self.stop_event = asyncio.Event()
        self.task = None
        self._running = False

    async def start(self):
        if self._running:
            return
        self._running = True
        self.stop_event.clear()
        self.channel = grpc.insecure_channel(cfg.get_yolo_service_address)
        self.stub = yolo_service_grpc.YoloServiceStub(self.channel)
        self.task = asyncio.create_task(self._prediction_loop())
        logger.info("prediction manager started")

    async def _prediction_loop(self):
        while not self.stop_event.is_set():
            await asyncio.sleep(0.033)
            frame = None
            async with self.camera.frame_lock:
                if self.camera.latest_frame is not None:
                    frame = self.camera.latest_frame.copy()
            if frame is None:
                continue

            try:
                _, encoded_image = cv2.imencode(".jpg", frame)
                image_bytes = encoded_image.tobytes()
                image_frame = yolo_service.ImageFrame()  # pyright: ignore
                image_frame.image_data = image_bytes
                image_frame.timestamp = int(time.time() * 1000)

                prediction = await asyncio.to_thread(self.stub.Predict, image_frame)  # pyright: ignore
                async with self.last_prediction_lock:
                    self.last_prediction = prediction.detections

            except grpc.RpcError as e:
                logger.error(f"gRPC error: {e}")
            except Exception as e:
                logger.error(f"Prediction request error: {e}")

    async def predict(self, frame):
        if not self._running:
            return None
        async with self.last_prediction_lock:
            return self.last_prediction

    async def stop(self):
        if not self._running:
            return

        self._running = False
        self.stop_event.set()

        if self.channel:
            self.channel.close()
        if self.task:
            try:
                await asyncio.wait_for(self.task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "prediction task did not finish within timeout, forcefully cancelling"
                )
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
            finally:
                if self.channel:
                    self.channel.close()
                self.channel = None
                self.stub = None
                self.task = None
        logger.info("prediction manager stopped")

    async def join(self):
        if self.task:
            try:
                await self.task
            except asyncio.CancelledError:
                pass
