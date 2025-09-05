#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run YOLOv8 training with CLI args
# Usage examples:
#   ./scripts/train_yolo.sh --data data/Peru_License_Plate/data.yaml --model yolov8n.yaml
#   ./scripts/train_yolo.sh --data data/LP-11k/data.yaml --model yolov8s.yaml --epochs 100 --batch 32 --imgsz 640 --device 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec python3 "${REPO_ROOT}/detectors/ultralytic_yolo/train_ultralytics.py" "$@"
