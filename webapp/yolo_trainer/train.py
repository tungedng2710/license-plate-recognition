#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import argparse
from datetime import datetime

# ───────────────────────── Dependencies ─────────────────────────
# pip install mlflow[extras]==2.13.0 psycopg2-binary minio ultralytics
import mlflow
from minio import Minio
from ultralytics import YOLO, settings
from ultralytics.yolo.engine.trainer import BaseTrainer

# ───────────────────────── Postgres ⟷ MLflow ────────────────────
# Environment variables give maximum flexibility when you move
# between dev / staging / prod.
PG_USER     = os.getenv("PG_USER",     "mlflowuser")
PG_PASSWORD = os.getenv("PG_PASSWORD", "mlflowpass")
PG_HOST     = os.getenv("PG_HOST",     "localhost")
PG_PORT     = int(os.getenv("PG_PORT", "5432"))
PG_DB       = os.getenv("PG_DB",       "mlflowdb")

# 👉 MLflow tracking URI that points **directly** at Postgres.
#    MLflow will manage all run / experiment metadata inside this DB.
TRACKING_URI = (
    f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
)
mlflow.set_tracking_uri(TRACKING_URI)

# If you prefer a dedicated MLflow tracking server, start it with:
# mlflow server \
#   --backend-store-uri postgresql://user:pwd@host:5432/dbname \
#   --default-artifact-root s3://ivamodels/mlflow \
#   --host 0.0.0.0 --port 7863
# …and switch TRACKING_URI to "http://0.0.0.0:7863".

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "yolo-ultralytics")
mlflow.set_experiment(EXPERIMENT_NAME)

# Tell Ultralytics to turn on its built-in MLflow callbacks
settings.update({"mlflow": True})

# ───────────────────────── MinIO defaults ───────────────────────
UPLOAD_RESULTS_TO_MINIO = os.getenv("UPLOAD_RESULTS_TO_MINIO", "true").lower() == "true"
MINIO_ENDPOINT    = os.getenv("MINIO_ENDPOINT", "0.0.0.0:9000")
MINIO_ACCESS_KEY  = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY  = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET      = os.getenv("MINIO_BUCKET", "ivamodels")
MINIO_PREFIX      = os.getenv("MINIO_PREFIX", "yolo_runs")

# ───────────────────────── Helpers ──────────────────────────────
def upload_folder_to_minio(client: Minio, bucket: str,
                           local_dir: Path, prefix: str = "") -> None:
    """Recursively upload `local_dir` to `<bucket>/<prefix>` in MinIO."""
    for root, _, files in os.walk(local_dir):
        for fname in files:
            fpath = Path(root, fname)
            object_name = str(Path(prefix, fpath.relative_to(local_dir))).replace("\\", "/")
            client.fput_object(bucket, object_name, str(fpath))

# ───────────────────────── Trainer class ────────────────────────
class ProgressCallback(BaseTrainer):
    def on_epoch_end(self, epoch, results):
        print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {results['loss']:.4f}")
        
class YOLOTrainer:
    def __init__(self, model_path: str, data_path: str):
        assert model_path, "model_path must not be empty"
        assert data_path,  "data_path must not be empty"

        # Connect to MinIO (optional ─ only if env flag is true)
        self.client = None
        if UPLOAD_RESULTS_TO_MINIO:
            try:
                self.client = Minio(
                    MINIO_ENDPOINT,
                    access_key=MINIO_ACCESS_KEY,
                    secret_key=MINIO_SECRET_KEY,
                    secure=False,
                )
                self.client.list_buckets()  # sanity check
                if not self.client.bucket_exists(MINIO_BUCKET):
                    self.client.make_bucket(MINIO_BUCKET)
                print(f"[MinIO] Connected to {MINIO_ENDPOINT}")
            except Exception as exc:
                print(f"[MinIO] Disabled → {exc}")

        self.model_path = model_path
        self.model = YOLO(model_path)
        self.data  = data_path

    # ────────────────────────────────────────────────────────────
    def train(self, **kwargs):
        """
        Forward any ultralytics.YOLO.train keyword via **kwargs
        (epochs, imgsz, batch, device, etc.).
        """
        # Optional auto-run-name
        if kwargs.pop("auto_set_name", False):
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_tag = Path(self.model_path).stem
            data_tag  = Path(self.data).parent.name or Path(self.data).stem
            kwargs["name"] = f"{model_tag}_{data_tag}_{kwargs.get('epochs', '??')}eps_" \
                             f"{kwargs.get('imgsz', '??')}_{ts}"

        # Start a new MLflow run (nested=False ⇢ top-level run)
        with mlflow.start_run(run_name=kwargs.get("name")) as run:
            # Log hyper-parameters
            mlflow.log_param("model_path", self.model_path)
            mlflow.log_param("data_path",  self.data)
            mlflow.log_params({k: v for k, v in kwargs.items() if v is not None})

            # Actual training
            results = self.model.train(data=self.data, callbacks=[ProgressCallback], **kwargs)

            # Ultralytics already streams per-epoch metrics to MLflow
            #     via its callback.  Here we log any final numbers we want.
            if results and hasattr(results, "metrics") and results.metrics:
                for k, v in results.metrics.items():
                    mlflow.log_metric(k, float(v))

            # Log full run directory as MLflow artifacts
            if results:
                run_dir = Path(results.save_dir)
                mlflow.log_artifacts(str(run_dir))

            # Optionally mirror to MinIO
            if self.client and results:
                prefix = f"{MINIO_PREFIX}/{run.info.run_id}"
                upload_folder_to_minio(self.client, MINIO_BUCKET, run_dir, prefix)
                print(f"Uploaded artifacts → s3://{MINIO_BUCKET}/{prefix}/")

            return results

# ───────────────────────── CLI boilerplate ──────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a YOLO model with MLflow + Postgres metadata + optional MinIO artifact sync"
    )

    # Required
    p.add_argument("--model-path", required=True,
                   help="Path to the YOLO model weights (e.g. ./weights/yolov8l.pt)")
    p.add_argument("--data-path",  required=True,
                   help="Ultralytics YAML dataset file (e.g. ./datasets/data.yaml)")

    # Common training knobs
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz",  type=int, default=640, help="Image size")
    p.add_argument("--batch",  type=int, default=32)
    p.add_argument("--device", default="0",
                   help="GPU device id(s) – 'cpu', '0', or '0,1'")

    # QoL toggles
    p.add_argument("--pretrained",  action="store_true", help="Use pretrained weights")
    p.add_argument("--resume",      action="store_true", help="Resume previous run")
    p.add_argument("--cache",       action="store_true", help="Cache images")
    p.add_argument("--cos-lr",      action="store_true", help="Use cosine LR schedule")
    p.add_argument("--auto-set-name", action="store_true",
                   help="Auto-generate run name with timestamp")

    return p.parse_args()

# ───────────────────────── Entrypoint ───────────────────────────
if __name__ == "__main__":
    args = parse_args()

    # Device arg massaging (int / list[int] / 'cpu')
    device_arg = args.device
    if "," in device_arg:
        device_arg = [int(d) for d in device_arg.split(",") if d.strip().isdigit()]
    elif device_arg.isdigit():
        device_arg = int(device_arg)

    trainer = YOLOTrainer(model_path=args.model_path,
                          data_path=args.data_path)

    trainer.train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device_arg,
        pretrained=args.pretrained,
        resume=args.resume,
        cache=args.cache,
        cos_lr=args.cos_lr,
        auto_set_name=args.auto_set_name,
    )