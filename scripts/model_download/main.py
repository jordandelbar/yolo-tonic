import os

from pathlib import Path

from ultralytics import YOLO


def main():
    base_path = Path(__file__).parents[0]
    onnx_path = base_path / "yolov8m.onnx"
    target_path = base_path.parents[1] / "yolo_prediction" / "models" / "yolov8m.onnx"

    if not onnx_path.exists():
        model = YOLO("yolov8m.pt")
        model.fuse()
        model.export(format="onnx", imgsz=[640, 640], simplify=True)
    else:
        print("yolov8 already downloaded")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    os.rename(onnx_path, target_path)


if __name__ == "__main__":
    main()
