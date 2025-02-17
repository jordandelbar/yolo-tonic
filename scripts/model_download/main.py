import os

from pathlib import Path

from ultralytics import YOLO


if not os.path.exists(f"{Path(__file__).parents[0]}/model_download/yolov8m.onnx"):
    model = YOLO("yolov8m.pt")
    model.export(format="onnx", imgsz=[640, 640])
else:
    print("yolov8 already downloaded")
