"""
Train Ultralytics YOLO with CLI arguments.

Examples:
  python detectors/ultralytic_yolo/train_yolov8.py \
    --data data/Peru_License_Plate/data.yaml \
    --model yolov8n.yaml --epochs 200 --batch 16 --imgsz 320 --device 0

You can also use the bash wrapper:
  ./scripts/train_yolov8.sh --data data/Peru_License_Plate/data.yaml --model yolov8n.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _resolve_path(p: str | None, repo_root: Path) -> str | None:
    if not p:
        return p
    path = Path(p)
    if path.is_absolute() or path.exists():
        return str(path)
    candidate = repo_root / p
    return str(candidate) if candidate.exists() else p


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Ultralytics YOLO model with custom parameters.")
    parser.add_argument("--model", default="yolov8n.yaml", help="Model config or weights file (e.g., yolov8n.yaml or yolov8n.pt)")
    parser.add_argument("--data", help="Path to data.yaml file")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=320, help="Image size")
    parser.add_argument("--device", default="0", help="Device to use, e.g. '0', '0,1', or 'cpu'")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--project", default=None, help="Project directory to save runs")
    parser.add_argument("--name", default=None, help="Run name")
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing project/name")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    args = _parse_args()

    model_path = _resolve_path(args.model, repo_root)
    data_path = _resolve_path(args.data, repo_root) if args.data else None

    if not data_path:
        # Attempt to auto-detect if a single candidate exists at data/*/data.yaml
        candidates = list((repo_root / "data").glob("*/data.yaml"))
        if len(candidates) == 1:
            data_path = str(candidates[0])
        else:
            raise SystemExit(
                "--data is required (found %d candidates under data/*/data.yaml)." % len(candidates)
            )

    # Import lazily so `-h/--help` works even if ultralytics is not installed yet
    from ultralytics import YOLO
    model = YOLO(model_path)

    train_kwargs = dict(
        data=data_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        resume=args.resume,
        verbose=args.verbose,
    )

    if args.project is not None:
        train_kwargs["project"] = args.project
    if args.name is not None:
        train_kwargs["name"] = args.name
    if args.exist_ok:
        train_kwargs["exist_ok"] = True
    if args.patience is not None:
        train_kwargs["patience"] = args.patience
    if args.seed is not None:
        train_kwargs["seed"] = args.seed

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
