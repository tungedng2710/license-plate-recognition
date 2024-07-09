"""
Train the YOLOv8 model
"""
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
model.train(data="/home/tungn197/code/license-plate-recognition/data/vn_plate_yolo/data.yaml",
            epochs=200,
            batch=16,
            imgsz=320,
            device=[0])