#!/bin/bash

SCRIPT_DIR=$(dirname "$0")

if [ ! -f "$SCRIPT_DIR/yolov8m.onnx" ]; then
  python3 -m venv "$SCRIPT_DIR/.venv"
  source "$SCRIPT_DIR/.venv/bin/activate"
  pip install -r "$SCRIPT_DIR/requirements.txt"
  python "$SCRIPT_DIR/main.py"
else
  echo "yolov8m.onnx exists. Skipping."
fi
