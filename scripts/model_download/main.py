import os

from pathlib import Path

from ultralytics import YOLO


def main():
    file_path = Path(__file__)
    base_path = file_path.parents[0]
    onnx_path = base_path / "yolov8m.onnx"
    pt_model = file_path / "yolov8m.pt"
    target_path = base_path.parents[1] / "yolo_prediction" / "models" / "yolov8m.onnx"

    if not onnx_path.exists():
        model = YOLO("yolov8m.pt")
        model.fuse()
        model.export(format="onnx", imgsz=[640, 640], simplify=True)
    else:
        print("yolov8 already downloaded")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    os.rename(onnx_path, target_path)
    if os.path.exists(pt_model):
        os.remove(pt_model)


if __name__ == "__main__":
    main()
