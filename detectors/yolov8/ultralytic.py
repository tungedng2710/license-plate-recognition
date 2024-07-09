"""
Ultralytics YOLOv8 model for object detection
"""
import torch
import numpy as np
import onnx
import onnxruntime as ort
from ultralytics import YOLO

class YOLOv8():
    def __init__(self, config):
        self.model = YOLO(config)
        self.device = torch.device("cuda:0")

    def train(self, data_yaml, epochs, batch, device):
        self.model.train(data=data_yaml,
                         epochs=epochs,
                         batch=batch,
                         device=device)

    def predict(self, image_path):
        metrics = self.model.val()
        return self.model(image_path)
    
    def export(self):
        self.model.info(verbose=True)
        self.model.export(format="onnx", dynamic=False, opset=12, batch=1)
        

if __name__ == "__main__":
    yolov8 = YOLOv8("/home/tungn197/code/license-plate-recognition/vehicle_yolov8s_640.pt")
    yolov8.export()

    # image = np.random.rand(1, 3, 640, 640).astype(np.float32)
    # ort_sess = ort.InferenceSession("../../weights/tuannb_best.onnx",
    #                                 providers=['CUDAExecutionProvider'])
    # output = ort_sess.run(None, {'images': image})
    # print(output[0].shape)

