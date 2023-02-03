from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
results = model.train(data="data/vehicle_yolov8/data.yaml", 
                      epochs=150,
                      batch=32)