from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel
from pathlib import Path
import subprocess
import re
from threading import Lock
import numpy as np
import cv2
import json
from types import SimpleNamespace
from test_alpr import ALPR

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "frontend"
REPO_ROOT = BASE_DIR.parent
TRAIN_SCRIPT = REPO_ROOT / "detectors" / "yolov9" / "train.py"
SYNC_SCRIPT = REPO_ROOT / "sync_with_minio.sh"

with open(REPO_ROOT / "minio_config.json") as f:
    MINIO_CFG = json.load(f)
MINIO_ENDPOINT = MINIO_CFG["endpoint"]
MINIO_ACCESS_KEY = MINIO_CFG["access_key"]
MINIO_SECRET_KEY = MINIO_CFG["secret_key"]
MINIO_BUCKET = MINIO_CFG["bucket"]

# Initialize ALPR model
opts = SimpleNamespace(
    vehicle_weight=str(REPO_ROOT / "weights" / "vehicle_yolov8s_640.pt"),
    plate_weight=str(REPO_ROOT / "weights" / "plate_yolov8n_320_2024.pt"),
    vconf=0.6,
    pconf=0.25,
    ocr_thres=0.8,
    device="cpu",
)
alpr_model = ALPR(opts)

class TrainRequest(BaseModel):
    dataset: str
    batch: int
    img_size: int
    model: str
    epochs: int

def list_datasets() -> list[str]:
    try:
        subprocess.run(
            ["/usr/local/bin/mc", "alias", "set", "local", MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY],
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            ["/usr/local/bin/mc", "ls", f"local/{MINIO_BUCKET}"],
            check=True,
            capture_output=True,
            text=True,
        )
        datasets = []
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if parts:
                name = parts[-1].rstrip('/')
                datasets.append(name)
        return datasets
    except Exception:
        return []

@app.get("/api/datasets")
def get_datasets():
    return {"datasets": list_datasets()}

def dataset_stats(dataset_name: str) -> dict:
    try:
        subprocess.run(
            ["/usr/local/bin/mc", "alias", "set", "local", MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY],
            check=True,
            capture_output=True,
        )
        stats: dict[str, int | list[str]] = {}
        for split in ["train", "val", "test"]:
            result = subprocess.run(
                ["/usr/local/bin/mc", "ls", f"local/{MINIO_BUCKET}/{dataset_name}/{split}/images"],
                check=True,
                capture_output=True,
                text=True,
            )
            lines = [line for line in result.stdout.splitlines() if line.strip()]
            stats[split] = len(lines)
            if split == "train":
                samples = [l.split()[-1] for l in lines][:12]
                stats["samples"] = [
                    f"{MINIO_ENDPOINT}/{MINIO_BUCKET}/{dataset_name}/train/images/{s}"
                    for s in samples
                ]
        return stats
    except Exception:
        return {"train": 0, "val": 0, "test": 0, "samples": []}


@app.get("/api/datasets/{dataset_name}/stats")
def get_dataset_stats(dataset_name: str):
    return dataset_stats(dataset_name)


training_progress = 0
training_running = False
progress_lock = Lock()


def set_progress(value: int, running: bool | None = None):
    global training_progress, training_running
    with progress_lock:
        training_progress = value
        if running is not None:
            training_running = running


@app.get("/api/progress")
def get_progress():
    with progress_lock:
        return {"progress": training_progress, "running": training_running}

def run_training(req: TrainRequest):
    dataset_yaml = REPO_ROOT / "datasets" / req.dataset / "data.yaml"
    subprocess.run(["bash", str(SYNC_SCRIPT), req.dataset], check=False)
    cfg = REPO_ROOT / "detectors" / "yolov9" / "models" / "detect" / f"yolov9-{req.model}.yaml"
    name = f"yolov9-{req.model}-{req.dataset}"
    cmd = [
        "python",
        str(TRAIN_SCRIPT),
        "--batch",
        str(req.batch),
        "--img",
        str(req.img_size),
        "--cfg",
        str(cfg),
        "--name",
        name,
        "--epochs",
        str(req.epochs),
        "--data",
        str(dataset_yaml),
        "--weights",
        "",
    ]
    set_progress(0, True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        match = re.search(r"(\d+)/(\d+)", line)
        if match:
            cur = int(match.group(1))
            total = int(match.group(2))
            set_progress(int(cur / total * 100))
    proc.wait()
    set_progress(100, False)

@app.post("/api/train")
def train(req: TrainRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_training, req)
    return {"status": "started"}


@app.post("/api/alpr")
async def alpr(file: UploadFile = File(...)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    alpr_model.vehicles = []
    result = alpr_model(img)
    _, buffer = cv2.imencode('.jpg', result)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
