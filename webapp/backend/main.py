from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import subprocess
import re
from threading import Lock

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "frontend"
REPO_ROOT = BASE_DIR.parent
TRAIN_SCRIPT = REPO_ROOT / "detectors" / "yolov9" / "train.py"
SYNC_SCRIPT = REPO_ROOT / "sync_with_minio.sh"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")

class TrainRequest(BaseModel):
    dataset: str
    batch: int
    img_size: int
    model: str
    epochs: int

def list_datasets() -> list[str]:
    try:
        subprocess.run(
            ["mc", "alias", "set", "local", "http://localhost:9000", "minioadmin", "minioadmin"],
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            ["mc", "ls", "local/ivadatasets"],
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
