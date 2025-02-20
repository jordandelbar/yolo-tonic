import cv2
import logging
import asyncio


logger = logging.getLogger(__name__)


class Camera:
    def __init__(self):
        self.cap = None
        self.latest_frame = None
        self.frame_lock = asyncio.Lock()

    async def initialize(self):
        self.cap = cv2.VideoCapture(0)
        for _ in range(10):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                break
            logger.info("waiting for camera to initialize")
            await asyncio.sleep(0.5)
        else:
            logger.error("failed to initialize camera")
            exit(1)

    async def frames(self):
        while self.cap is not None and True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("failed to capture frame, skipping")
                await asyncio.sleep(0.1)
                continue
            async with self.frame_lock:
                self.latest_frame = frame.copy()
                yield frame

    async def release(self):
        if self.cap:
            self.cap.release()
