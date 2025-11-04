#!/usr/bin/env python3
"""
Generate a Markdown model zoo summary from Ultralytics YOLO run folders.

Example:
    python scripts/generate_model_zoo.py \
        --runs-dir runs/detect \
        --output MODEL_ZOO.md
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Markdown model zoo summary from YOLO run folders."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs/detect"),
        help="Path to the YOLO runs directory (defaults to runs/detect).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("MODEL_ZOO.md"),
        help="Destination Markdown file (defaults to MODEL_ZOO.md).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Project root used to make paths relative (defaults to repository root).",
    )
    return parser.parse_args()


def safe_float(value: Optional[str]) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def load_yaml(path: Path) -> Dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def parse_results_csv(path: Path) -> Tuple[Optional[Dict[str, str]], List[str]]:
    """
    Returns the dict representing the best epoch (by mAP50) and warnings.
    """
    warnings: List[str] = []
    if not path.is_file():
        warnings.append(f"missing results.csv")
        return None, warnings

    best_row: Optional[Dict[str, str]] = None
    best_map50 = -math.inf
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            warnings.append("results.csv has no header")
            return None, warnings
        for row in reader:
            map50 = safe_float(row.get("metrics/mAP50(B)"))
            if math.isnan(map50):
                continue
            if best_row is None or map50 > best_map50:
                best_row = row
                best_map50 = map50

    if best_row is None:
        warnings.append("no valid metrics/mAP50(B) rows")
    return best_row, warnings


def format_float(value: float, decimals: int = 3) -> str:
    if math.isnan(value):
        return "N/A"
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(value)


def relativize(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, FileNotFoundError):
        return path.as_posix()


def gather_run_summary(run_dir: Path, project_root: Path) -> Tuple[Optional[Dict], List[str]]:
    warnings: List[str] = []

    args_data = load_yaml(run_dir / "args.yaml")
    if not args_data:
        warnings.append("missing args.yaml")

    best_row, result_warnings = parse_results_csv(run_dir / "results.csv")
    warnings.extend(result_warnings)
    if best_row is None:
        return None, warnings

    def get_arg(key: str, default: str = "N/A") -> str:
        value = args_data.get(key) if isinstance(args_data, dict) else None
        if value is None:
            return default
        return str(value)

    best_map50 = safe_float(best_row.get("metrics/mAP50(B)"))
    best_map5095 = safe_float(best_row.get("metrics/mAP50-95(B)"))
    precision = safe_float(best_row.get("metrics/precision(B)"))
    recall = safe_float(best_row.get("metrics/recall(B)"))
    epoch = get_arg("epochs", "N/A")

    best_epoch = best_row.get("epoch", "N/A")
    data_path_str = get_arg("data")
    data_path = Path(data_path_str).expanduser() if data_path_str != "N/A" else None
    model_config = get_arg("model")
    imgsz = get_arg("imgsz")
    batch = get_arg("batch")

    summary = {
        "run_name": run_dir.name,
        "model": model_config,
        "data": relativize(data_path, project_root) if data_path else data_path_str,
        "imgsz": imgsz,
        "batch": batch,
        "epochs": epoch,
        "best_epoch": best_epoch,
        "map50": best_map50,
        "map5095": best_map5095,
        "precision": precision,
        "recall": recall,
    }
    return summary, warnings


def build_markdown(
    summaries: List[Dict],
    skipped: Dict[str, List[str]],
    warnings_map: Dict[str, List[str]],
    runs_dir: Path,
) -> str:
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    header = [
        "# YOLO Model Zoo",
        "",
        f"- Generated: {generated_at}",
        f"- Source runs directory: `{runs_dir.as_posix()}`",
        "",
    ]

    if summaries:
        table_header = (
            "| Run | Model | Data | Img | Batch | Epochs | Best Epoch | mAP50 | mAP50-95 | Precision | Recall |"
        )
        table_sep = (
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        )

        rows = []
        for entry in summaries:
            row = "| {run_name} | `{model}` | `{data}` | {imgsz} | {batch} | {epochs} | {best_epoch} | {map50} | {map5095} | {precision} | {recall} |".format(
                run_name=entry["run_name"],
                model=entry["model"],
                data=entry["data"],
                imgsz=entry["imgsz"],
                batch=entry["batch"],
                epochs=entry["epochs"],
                best_epoch=entry["best_epoch"],
                map50=format_float(entry["map50"]),
                map5095=format_float(entry["map5095"]),
                precision=format_float(entry["precision"]),
                recall=format_float(entry["recall"]),
            )
            rows.append(row)

        header.extend([table_header, table_sep, *rows, ""])

    if warnings_map:
        header.append("## Warnings")
        for run_name, reasons in warnings_map.items():
            header.append(f"- `{run_name}`: {', '.join(reasons)}")
        header.append("")

    if skipped:
        header.append("## Skipped Runs")
        for run_name, reasons in skipped.items():
            header.append(f"- `{run_name}`: {', '.join(reasons)}")
        header.append("")

    return "\n".join(header)


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir
    project_root = args.project_root

    if not runs_dir.is_dir():
        raise SystemExit(f"Runs directory does not exist: {runs_dir}")

    summaries: List[Dict] = []
    skipped: Dict[str, List[str]] = {}
    warnings_map: Dict[str, List[str]] = {}
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary, warnings = gather_run_summary(run_dir, project_root)
        if summary is None:
            skipped[run_dir.name] = warnings or ["no usable metrics found"]
            continue
        summaries.append(summary)
        if warnings:
            warnings_map[run_dir.name] = warnings

    summaries.sort(
        key=lambda item: item["map50"] if not math.isnan(item["map50"]) else float("-inf"),
        reverse=True,
    )

    markdown = build_markdown(summaries, skipped, warnings_map, runs_dir)
    output_path: Path = args.output
    output_path.write_text(markdown, encoding="utf-8")

    print(f"Model zoo written to {output_path}")


if __name__ == "__main__":
    main()
