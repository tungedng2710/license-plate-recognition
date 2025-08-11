from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel
from pathlib import Path
import subprocess
import re
import os
from threading import Lock
from multiprocessing import Process
import numpy as np
import cv2
import json
import yaml
import zipfile
import shutil
from types import SimpleNamespace
from test_alpr import ALPR
from backend.yolo_trainer.train import YOLOTrainer

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "frontend"
REPO_ROOT = BASE_DIR.parent
TRAIN_SCRIPT = BASE_DIR / "yolo_trainer" / "train.py"
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
        stats = {}
        for split in ["train", "val", "test"]:
            result = subprocess.run(
                ["/usr/local/bin/mc", "ls", f"local/{MINIO_BUCKET}/{dataset_name}/{split}/images"],
                check=True,
                capture_output=True,
                text=True,
            )
            count = len([line for line in result.stdout.splitlines() if line.strip()])
            stats[split] = count

        # Retrieve class names from data.yaml if available
        try:
            yaml_res = subprocess.run(
                ["/usr/local/bin/mc", "cat", f"local/{MINIO_BUCKET}/{dataset_name}/data.yaml"],
                check=True,
                capture_output=True,
                text=True,
            )
            data_yaml = yaml.safe_load(yaml_res.stdout)
            names = data_yaml.get("names", []) if isinstance(data_yaml, dict) else []
            stats["classes"] = len(names)
            stats["tags"] = names
        except Exception:
            stats["classes"] = 0
            stats["tags"] = []

        return stats
    except Exception:
        return {"train": 0, "val": 0, "test": 0, "classes": 0, "tags": []}


@app.get("/api/datasets/{dataset_name}/stats")
def get_dataset_stats(dataset_name: str):
    return dataset_stats(dataset_name)


@app.get("/api/datasets/{dataset_name}/thumbnail")
def get_dataset_thumbnail(dataset_name: str):
    try:
        subprocess.run(
            ["/usr/local/bin/mc", "alias", "set", "local", MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY],
            check=True,
            capture_output=True,
        )
        ls_res = subprocess.run(
            ["/usr/local/bin/mc", "ls", f"local/{MINIO_BUCKET}/{dataset_name}/train/images"],
            check=True,
            capture_output=True,
            text=True,
        )
        first_file = None
        for line in ls_res.stdout.splitlines():
            parts = line.strip().split()
            if parts:
                first_file = parts[-1]
                break
        if not first_file:
            return Response(status_code=404)
        cat_res = subprocess.run(
            ["/usr/local/bin/mc", "cat", f"local/{MINIO_BUCKET}/{dataset_name}/train/images/{first_file}"],
            check=True,
            capture_output=True,
        )
        suffix = Path(first_file).suffix.lower()
        media_type = "image/png" if suffix == ".png" else "image/jpeg"
        return Response(content=cat_res.stdout, media_type=media_type)
    except Exception:
        return Response(status_code=404)


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    tmp_dir = Path("/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip = tmp_dir / file.filename
    with open(tmp_zip, "wb") as f:
        f.write(await file.read())

    extract_dir = tmp_dir / Path(file.filename).stem
    with zipfile.ZipFile(tmp_zip, "r") as z:
        z.extractall(extract_dir)

    try:
        subprocess.run(
            ["/usr/local/bin/mc", "alias", "set", "local", MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["/usr/local/bin/mc", "cp", "--recursive", str(extract_dir), f"local/{MINIO_BUCKET}/{extract_dir.name}"],
            check=True,
            capture_output=True,
        )
    except Exception:
        pass
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)
        try:
            tmp_zip.unlink()
        except Exception:
            pass

    return {"status": "uploaded"}


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

def _train_worker(model_path: str, dataset_yaml: Path, epochs: int, img_size: int, batch: int) -> None:
    trainer = YOLOTrainer(model_path=str(model_path), data_path=str(dataset_yaml))
    trainer.train(
        epochs=int(epochs),
        imgsz=int(img_size),
        batch=int(batch),
        device="0",
        pretrained=False,
        resume=False,
        cache=False,
        cos_lr=False,
        auto_set_name=True,
    )


def run_training(req: TrainRequest):
    dataset_yaml = REPO_ROOT / "data" / req.dataset / "data.yaml"
    model_path = f"{BASE_DIR}/yolo_trainer/weights/{req.model}.pt"

    set_progress(0, True)
    p = Process(
        target=_train_worker,
        args=(model_path, dataset_yaml, req.epochs, req.img_size, req.batch),
    )
    p.start()
    p.join()
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
