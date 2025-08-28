from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, Response
from pathlib import Path
import cv2
import numpy as np
import json
from types import SimpleNamespace
from test_alpr import ALPR

app = FastAPI()

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


def gen_frames(url: str, process: bool = False):
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if process:
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
def alpr_stream(url: str):
    return StreamingResponse(gen_frames(url, True), media_type="multipart/x-mixed-replace; boundary=frame")


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


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
