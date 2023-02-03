import cv2
import shutil
import os
import random
import string
import torch.nn as nn

from ultralytics import YOLO
from PIL import Image

def draw_text(img, text,
            pos=(0, 0),
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=1,
            font_thickness=2,
            text_color=(0, 0, 255),
            text_color_bg=(0, 255, 0)
            ):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        print("This is a dummy OCR model, replace it with another one!")

    def forward(self, x):
        number = random.uniform(1, 9)
        number = int(10000*number)
        number = str(number)
        letter = random.choice(string.ascii_uppercase)
        dummy_output = "30"+letter+number
        return dummy_output

class Pipeline():
    def __init__(self,
                 source: str = "data",
                 vehicle_weight: str = None,
                 plate_weight: str = None,
                 use_sd_resolution: bool = False,
                 use_hd_resolution: bool = True,
                 use_fhd_resolution: bool = False):
        self.source = source
        self.vehicle_weight = vehicle_weight
        self.image = cv2.imread(source)
        self.vehicle_model = YOLO(vehicle_weight)
        self.plate_model = YOLO(plate_weight)
        self.ocr_model = DummyModel()

        # Resolution config for displaying
        self.use_sd_resolution = use_hd_resolution
        self.use_hd_resolution = use_hd_resolution
        self.use_fhd_resolution = use_fhd_resolution
        if use_hd_resolution and use_fhd_resolution:
            self.use_fhd_resolution = False
        self.use_sd_resolution = use_sd_resolution
        if use_sd_resolution:
            self.use_hd_resolution = False
            self.use_fhd_resolution = False

    def set_resolution(self, image):
        height, width, _ = image.shape
        ratio = width / height
        if self.use_sd_resolution:
            image = cv2.resize(image, (640, int(640 / ratio)))
        elif self.use_hd_resolution:
            image = cv2.resize(image, (1280, int(1280 / ratio)))
        elif self.use_fhd_resolution:
            image = cv2.resize(image, (1920, int(1920 / ratio)))
        else:
            pass
        return image

    def ocr(self, image):
        text = self.ocr_model(image)
        return text

    def run(self):
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FPS, int(30))
        if (cap.isOpened()== False):
            print("Error opening video file")
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                vehicle_results = self.vehicle_model(source=frame, save=False, conf=0.5)
                vehicle_boxes = vehicle_results[0].boxes.xyxy

                bgr_blue = (255, 0, 0)
                bgr_green = (0, 255, 0)
                bgr_red = (0, 0, 255)
                bgr_amber = (0, 191, 255)

                for box in vehicle_boxes:
                    if box is None:
                        continue
                    box = box.cpu().numpy().astype(int)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), bgr_green, thickness=2)

                    im = frame[box[1]:box[3], box[0]:box[2], :]
                    plate_results = self.plate_model(source=im, save=False, conf=0.1)
                    plate_boxes = plate_results[0].boxes.xyxy

                    for plate_box in plate_boxes:
                        if plate_box is None:
                            continue
                        plate_box = plate_box.cpu().numpy().astype(int)
                        src_point = (plate_box[0]+box[0], plate_box[1]+box[1])
                        dst_point = (plate_box[2]+box[0], plate_box[3]+box[1])
                        cv2.rectangle(frame, src_point, dst_point, bgr_red, thickness=2)

                    ocr_text = self.ocr(im)
                    pos = (box[0], box[1]-5)
                    draw_text(frame, ocr_text, pos)
                
                frame = self.set_resolution(frame)
                cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            
if __name__ == "__main__":
    pipeline = Pipeline(source="data/TuKy.mp4",
                        vehicle_weight="weights/vehicle_yolov8.pt",
                        plate_weight="weights/plate_yolov8.pt",
                        use_sd_resolution=True)
    pipeline.run()