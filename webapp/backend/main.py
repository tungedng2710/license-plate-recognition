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

# Keep this last so it doesn't intercept /api/* routes
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
