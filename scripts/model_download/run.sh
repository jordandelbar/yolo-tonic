#!/bin/bash

SCRIPT_DIR=$(dirname "$0")

if [ ! -f "$SCRIPT_DIR/yolov8m.onnx" ]; then
  uv run "$SCRIPT_DIR/main.py"
else
  echo "yolov8m.onnx exists. Skipping."
fi
