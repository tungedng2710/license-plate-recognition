from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, Response
from pathlib import Path
import cv2
import numpy as np
import json
from types import SimpleNamespace
from typing import Optional
from test_alpr import ALPR
import platform
import subprocess
import shutil
import os
from typing import List
try:
    import psutil  # optional dependency
except Exception:  # pragma: no cover
    psutil = None
import requests

app = FastAPI()

# Allow cross-origin usage of the API (useful when serving frontend elsewhere)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "frontend"
REPO_ROOT = BASE_DIR.parent
RTSP_FILE = BASE_DIR / "rtsp_url.json"

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


def gen_frames(url: str, process: bool = False, vconf: Optional[float] = None, pconf: Optional[float] = None):
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if process:
            # Update thresholds dynamically if provided
            if vconf is not None:
                alpr_model.opts.vconf = float(vconf)
            if pconf is not None:
                alpr_model.opts.pconf = float(pconf)
            alpr_model.vehicles = []
            frame = alpr_model(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()


@app.get("/api/video")
def video_stream(url: str):
    return StreamingResponse(gen_frames(url, False), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/alpr_stream")
def alpr_stream(url: str, vconf: Optional[float] = None, pconf: Optional[float] = None):
    return StreamingResponse(
        gen_frames(url, True, vconf=vconf, pconf=pconf),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/rtsp_urls")
def get_rtsp_urls():
    try:
        with open(RTSP_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


@app.post("/api/alpr")
async def alpr(file: UploadFile = File(...)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    alpr_model.vehicles = []
    result = alpr_model(img)
    _, buffer = cv2.imencode('.jpg', result)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


 # Serve frontend after registering API routes so it doesn't shadow them


# Chatbot streaming proxy to Ollama-compatible API (local Ollama)
DEFAULT_OLLAMA_URL = "http://0.0.0.0:7860/api/generate"


@app.post("/api/chat")
def chat(payload: dict):
    prompt: str = payload.get("prompt", "").strip()
    model: Optional[str] = payload.get("model") or "tonai_chat"
    url: str = payload.get("url") or DEFAULT_OLLAMA_URL
    image_path: Optional[str] = payload.get("image_path")

    if not prompt:
        return Response(content="Prompt is required", status_code=400)

    def stream():
        req_payload = {"model": model, "prompt": prompt}
        if image_path:
            try:
                with open(image_path, "rb") as f:
                    import base64
                    req_payload["images"] = [base64.b64encode(f.read()).decode("utf-8")]
            except Exception:
                pass
        try:
            with requests.post(url, json=req_payload, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                    except Exception:
                        chunk = ""
                    if chunk:
                        yield chunk
        except Exception as e:
            # Gracefully stream an error message instead of 500ing the request
            msg = f"[chatbot error] {str(e)}"
            yield msg

    return StreamingResponse(stream(), media_type="text/plain")


@app.get("/api/chat")
def chat_info():
    """Friendly GET handler to avoid 405 when opening /api/chat in a browser."""
    message = (
        "Chat endpoint is ready. Use POST with JSON to /api/chat: "
        '{"prompt": "your message", "model": "tonai_chat"}'
    )
    return Response(content=message, media_type="text/plain")


@app.head("/api/chat")
def chat_head():
    return Response()


@app.get("/api/system_info")
def system_info():
    """Return basic system information (OS, CPU, RAM, GPU).

    Uses optional psutil if available; otherwise falls back to /proc and platform.
    """
    info = {}

    # OS and Python
    info["os"] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }

    # CPU
    cpu = {
        "cores_logical": os.cpu_count() or 0,
    }
    # Try to get CPU model name (Linux)
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        cpu["model"] = line.split(":", 1)[1].strip()
                        break
    except Exception:
        pass
    # Physical cores via psutil if present
    try:
        if psutil is not None:
            cpu["cores_physical"] = psutil.cpu_count(logical=False)
            cpu["utilization_percent"] = psutil.cpu_percent(interval=0.1)
    except Exception:
        pass
    info["cpu"] = cpu

    # RAM
    mem = {}
    try:
        if psutil is not None:
            vm = psutil.virtual_memory()
            mem = {
                "total_gb": round(vm.total / (1024**3), 2),
                "available_gb": round(vm.available / (1024**3), 2),
                "used_gb": round(vm.used / (1024**3), 2),
                "percent": vm.percent,
            }
        else:
            # Fallback for Linux
            if platform.system() == "Linux":
                total_kb = available_kb = None
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            total_kb = int(line.split()[1])
                        elif line.startswith("MemAvailable:"):
                            available_kb = int(line.split()[1])
                if total_kb:
                    mem["total_gb"] = round(total_kb / (1024**2), 2)
                if available_kb is not None and total_kb:
                    mem["available_gb"] = round(available_kb / (1024**2), 2)
                    mem["used_gb"] = round((total_kb - available_kb) / (1024**2), 2)
    except Exception:
        pass
    info["ram"] = mem

    # GPU via nvidia-smi if available
    gpus: List[dict] = []
    if shutil.which("nvidia-smi"):
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ]
            out = subprocess.check_output(cmd, text=True, timeout=2)
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    name, mem_total, mem_used, driver = parts[:4]
                    gpus.append({
                        "name": name,
                        "memory_total_mb": int(float(mem_total)),
                        "memory_used_mb": int(float(mem_used)),
                        "driver": driver,
                    })
        except Exception:
            pass
    info["gpus"] = gpus

    return info

# Keep this last so it doesn't intercept /api/* routes
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
