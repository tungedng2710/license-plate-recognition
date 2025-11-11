"""
Lightweight FastAPI microservice used by parking systems to read license plates
from images that are uploaded as base64 strings.

Usage:
    uvicorn scripts.parking_services:app --reload --host 0.0.0.0 --port 8080

POST /api/plates
----------------
Request body (JSON):
{
    "image_base64": "<base64 image>",
    "conf_threshold": 0.3,      # optional detector confidence in [0, 1]
    "ocr_threshold": 0.6        # optional OCR confidence in [0, 1]
}

Response body (JSON):
{
    "plate_count": 1,
    "plates": [
        {
            "bbox": {"x1": 10, "y1": 20, "x2": 200, "y2": 120},
            "detection_confidence": 0.87,
            "ocr_text": "51F12345",
            "ocr_confidence": 0.91,
            "is_legit": true
        }
    ]
}
"""
from __future__ import annotations

import base64
import binascii
import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException

try:  # Pydantic v2 first
    from pydantic import BaseModel, Field, field_validator as _validator
except ImportError:  # pragma: no cover
    from pydantic import BaseModel, Field, validator as _validator

from starlette.concurrency import run_in_threadpool
from ultralytics import YOLO
from paddleocr import PaddleOCR

from utils.utils import check_legit_plate, crop_expanded_plate


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHT_PATH = os.environ.get(
    "PLATE_MODEL_PATH",
    str(REPO_ROOT / "weights" / "license_plate_detector.pt"),
)
DEFAULT_DEVICE = os.environ.get("PLATE_DEVICE", "cpu")
DEFAULT_DET_CONF = float(os.environ.get("PLATE_CONF_THRESHOLD", 0.25))
DEFAULT_OCR_CONF = float(os.environ.get("PLATE_OCR_THRESHOLD", 0.8))
DEFAULT_IMAGE_SIZE = int(os.environ.get("PLATE_IMG_SIZE", 640))
DEFAULT_CROP_EXPAND_RATIO = float(os.environ.get("PLATE_CROP_EXPAND_RATIO", 0.25))


class PlateRequest(BaseModel):
    image_base64: str = Field(..., description="Image encoded as base64 string.")
    conf_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Detector confidence threshold."
    )
    ocr_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="OCR confidence threshold."
    )

    @_validator("image_base64")
    def _not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("image_base64 cannot be empty")
        return value


class PlateBBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class PlateResponseItem(BaseModel):
    bbox: PlateBBox
    detection_confidence: float
    ocr_text: str
    ocr_confidence: float
    is_legit: bool


class PlateResponse(BaseModel):
    plate_count: int
    plates: List[PlateResponseItem]


@dataclass
class DetectionResult:
    bbox: PlateBBox
    score: float
    text: str
    text_conf: float
    legit: bool


@dataclass
class TextResult:
    text: str
    confidence: float
    legit: bool


def _resolve_device(requested: str) -> str:
    requested = (requested or "").strip().lower()
    if requested in {"", "auto"}:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if requested.isdigit():
        return f"cuda:{requested}" if torch.cuda.is_available() else "cpu"
    return requested


class PlatePipeline:
    """
    Wraps YOLO-based plate detection and PaddleOCR recognition.
    Serializes inference with a lock because the underlying models are
    not guaranteed to be thread-safe.
    """

    def __init__(
        self,
        *,
        weight_path: str,
        device: str,
        det_conf: float,
        ocr_conf: float,
        imgsz: int = 640,
        crop_expand_ratio: float = 0.25,
    ) -> None:
        self.weight_path = Path(weight_path)
        if not self.weight_path.exists():
            raise FileNotFoundError(
                f"Plate detector weight not found: {self.weight_path}"
            )
        self.device = _resolve_device(device)
        self.imgsz = imgsz
        self.default_det_conf = det_conf
        self.default_ocr_conf = ocr_conf
        self.crop_expand_ratio = crop_expand_ratio
        self._lock = Lock()

        self.detector = YOLO(str(self.weight_path), task="detect")
        try:
            self.detector.to(self.device)
        except Exception:
            pass

        ocr_kwargs: Dict[str, object] = dict(
            lang="en",
            # text_detection_model_name="PP-OCRv5_mobile_det",
            # text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        if self.device.startswith("cuda"):
            ocr_kwargs["device"] = "cpu"
        self.ocr = PaddleOCR(**ocr_kwargs)

    def _predict_text(self, plate_image: np.ndarray) -> TextResult:
        if plate_image is None or plate_image.size == 0:
            return TextResult(text="", confidence=0.0, legit=False)

        results = self.ocr.predict(input=plate_image)
        if not results:
            return TextResult(text="", confidence=0.0, legit=False)

        rec_texts = results[0].get("rec_texts", [])
        rec_scores = results[0].get("rec_scores", [])
        text = " ".join(rec_texts) if rec_texts else ""
        text = re.sub(r"[^A-Za-z0-9\-.]", "", text)
        if text and len(text) > 2 and text[0].isalpha() and text[2] == "C":
            text = text[:2] + "0" + text[3:]
        conf = float(sum(rec_scores) / len(rec_scores)) if rec_scores else 0.0
        return TextResult(
            text=text,
            confidence=conf,
            legit=bool(text and check_legit_plate(text)),
        )

    def __call__(
        self,
        image: np.ndarray,
        *,
        det_conf: Optional[float] = None,
        ocr_conf: Optional[float] = None,
    ) -> List[DetectionResult]:
        det_conf = float(det_conf if det_conf is not None else self.default_det_conf)
        ocr_conf = float(ocr_conf if ocr_conf is not None else self.default_ocr_conf)

        with self._lock:
            pred = self.detector(
                image,
                verbose=False,
                conf=det_conf,
                imgsz=self.imgsz,
                device=self.device,
            )[0]

            boxes = pred.boxes
            if boxes is None or len(boxes) == 0:
                return []

            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()

            results: List[DetectionResult] = []
            height, width = image.shape[:2]

            for bbox_arr, score in zip(xyxy, scores):
                x1, y1, x2, y2 = bbox_arr.astype(int).tolist()
                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width - 1))
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height - 1))

                bbox = PlateBBox(x1=x1, y1=y1, x2=x2, y2=y2)
                plate_crop = crop_expanded_plate(
                    (x1, y1, x2, y2),
                    image,
                    expand_ratio=self.crop_expand_ratio,
                )
                text_result = self._predict_text(plate_crop)
                text = text_result.text if text_result.confidence >= ocr_conf else ""
                legit = bool(text and text_result.legit and text_result.confidence >= ocr_conf)

                results.append(
                    DetectionResult(
                        bbox=bbox,
                        score=float(score),
                        text=text,
                        text_conf=text_result.confidence,
                        legit=legit,
                    )
                )

            return results


def decode_base64_image(payload: str) -> np.ndarray:
    """
    Decode a base64-encoded image. Supports optional data URL prefixes.
    """
    if "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        binary = base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {exc}") from exc

    np_buf = np.frombuffer(binary, dtype=np.uint8)
    image = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Unable to decode image bytes")
    return image


pipeline = PlatePipeline(
    weight_path=DEFAULT_WEIGHT_PATH,
    device=DEFAULT_DEVICE,
    det_conf=DEFAULT_DET_CONF,
    ocr_conf=DEFAULT_OCR_CONF,
    imgsz=DEFAULT_IMAGE_SIZE,
    crop_expand_ratio=DEFAULT_CROP_EXPAND_RATIO,
)

app = FastAPI(title="Parking Service", version="0.1.0")


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/plates")
async def read_plate(payload: PlateRequest) -> Dict[str, object]:
    image = decode_base64_image(payload.image_base64)

    def _run():
        return pipeline(
            image,
            det_conf=payload.conf_threshold,
            ocr_conf=payload.ocr_threshold,
        )

    detections = await run_in_threadpool(_run)

    plate_items = [
        PlateResponseItem(
            bbox=det.bbox,
            detection_confidence=det.score,
            ocr_text=clean(det.text),
            ocr_confidence=det.text_conf,
            is_legit=det.legit,
        )
        for det in detections
    ]

    response = PlateResponse(
        plate_count=len(plate_items),
        plates=plate_items,
    )

    # Maintain compatibility across Pydantic versions
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return response.dict()


def clean(text):
    cleaned_text = re.sub(r'honda|yamaha|vie', '', text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "scripts.parking_services:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8080)),
        reload=bool(int(os.environ.get("RELOAD", "0"))),
    )
