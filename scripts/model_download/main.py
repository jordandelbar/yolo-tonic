import os
from ultralytics import YOLO

if not os.path.exists("./yolov8m.onnx"):
    model = YOLO("yolov8m.pt")
    model.export(format="onnx", imgsz=[640, 640])
