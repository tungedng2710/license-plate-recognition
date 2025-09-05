"""
Ultralytics YOLO Exporter (ONNX, TensorRT)

This module provides a simple API and CLI to export Ultralytics
YOLO models to ONNX or TensorRT engine files. Training and
inference utilities have been intentionally removed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def _normalize_format(fmt: str) -> str:
    f = fmt.lower()
    if f in ("onnx",):
        return "onnx"
    if f in ("tensorrt", "engine", "trt"):
        return "engine"
    raise ValueError(f"Unsupported export format: {fmt}. Use 'onnx' or 'tensorrt'.")


def export_ultralytics(weights: str,
                       fmt: str = "onnx",
                       imgsz: int = 640,
                       dynamic: bool = True,
                       opset: int = 12,
                       device: str | None = None) -> None:
    """
    Export an Ultralytics YOLO model to ONNX or TensorRT.

    - weights: path to model weights (e.g., .pt)
    - fmt: 'onnx' or 'tensorrt'
    - imgsz: export image size (int or square)
    - dynamic: enable dynamic shapes for ONNX
    - opset: ONNX opset version
    - device: device string like 'cpu', '0', '0,1' (optional)
    """
    out_format = _normalize_format(fmt)
    model = YOLO(weights)

    export_kwargs = dict(format=out_format, imgsz=imgsz)
    if out_format == "onnx":
        export_kwargs.update(dynamic=dynamic, opset=opset)
    if device:
        export_kwargs.update(device=device)

    model.export(**export_kwargs)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Ultralytics YOLO to ONNX or TensorRT")
    p.add_argument("--weights", required=True, help="Path to model weights (.pt)")
    p.add_argument("--format", default="onnx", help="Export format: onnx | tensorrt")
    p.add_argument("--imgsz", type=int, default=640, help="Export image size (square)")
    p.add_argument("--dynamic", action="store_true", help="Enable dynamic shapes (ONNX)")
    p.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    p.add_argument("--device", default=None, help="Device: cpu | 0 | 0,1 (optional)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    weights_path = str(Path(args.weights))
    export_ultralytics(
        weights=weights_path,
        fmt=args.format,
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        opset=args.opset,
        device=args.device,
    )


if __name__ == "__main__":
    main()
