"""
Ultralytics YOLOv8 model for object detection
"""
import numpy as np
import onnx
import onnxruntime as ort
from ultralytics import YOLO

class YOLOv8():
    def __init__(self, config):
        self.model = YOLO(config)

    def train(self, data_yaml, epochs, batch, device):
        self.model.train(data=data_yaml,
                         epochs=epochs,
                         batch=batch,
                         device=device)

    def predict(self, image_path):
        metrics = self.model.val()
        return self.model(image_path)
    
    def export_to_onnx(self):
        # self.model.fuse()
        self.model.info(verbose=True)
        self.model.export(format="onnx", opset=12)
        

if __name__ == "__main__":
    # yolov8 = YOLOv8("weights/vehicle_yolov8n_1088.pt")
    # yolov8.export_to_onnx()

    image = np.random.rand(1, 3, 1088, 1088).astype(np.float32)
    ort_sess = ort.InferenceSession("../weights/vehicle_yolov8n_1088.onnx")
    output = ort_sess.run(None, {'images': image})
    print(output[0].shape)

