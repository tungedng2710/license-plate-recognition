import shutil
import os
import random
import string
import numpy as np
import cv2

import torch
from torch import nn

from ultralytics import YOLO

os.environ["YOLOv8_VERBOSE"] = "False"
BGR_COLORS = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "amber": (0, 191, 255)
}
def check_image_size(image, w_thres, h_thres):
    """
    Ignore small images
    Args: image, w_thres, h_thres
    """
    if w_thres is None:
        w_thres = 64
    if h_thres is None:
        h_thres = 64
    width, height, _ = image.shape
    if (width >= w_thres) and (height >= h_thres):
        return True
    else:
        return False

def draw_text(img, text,
            pos=(0, 0),
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=1,
            font_thickness=2,
            text_color=(0, 0, 255),
            text_color_bg=(0, 255, 0)
            ):
    """
    Minor modification of cv2.putText to add background color
    """
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    pos = (x, y + text_h + font_scale - 1)
    cv2.putText(img, text, pos, font, font_scale, text_color, font_thickness)

class DummyModel(nn.Module):
    """
    I'm just a Dummy model for filling the gap
    Replace me with an OCR model
    """
    def __init__(self):
        super().__init__()
        print("You are using dummy OCR model!")

    def forward(self, input):
        number = random.uniform(1, 9)
        number = int(10000*number)
        number = str(number)
        letter = random.choice(string.ascii_uppercase)
        dummy_output = "30"+letter+number
        return dummy_output

class Pipeline():
    """
    License plate OCR pipeline
    """
    def __init__(self,
                 source: str = "data",
                 vehicle_weight: str = None,
                 plate_weight: str = None,
                 use_sd_resolution: bool = False,
                 use_hd_resolution: bool = True,
                 use_fhd_resolution: bool = False,
                 save_plate_image: bool = False):
        # Pipeline core
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

        # Miscellaneous for displaying
        self.color = BGR_COLORS
        self.save_plate_image = save_plate_image
        if save_plate_image:
            vid_name = os.path.basename(self.source).split('.')[0]
            self.saved_path = "data/{vid_name}".format(vid_name=vid_name)
            if os.path.exists(self.saved_path):
                shutil.rmtree(self.saved_path)
            os.makedirs(self.saved_path)

    def set_resolution(self, image):
        """
        Set video resolution for displaying only.
        Note: this function doesn't config the input video of the model.
        Arg:
            image (OpenCV image): video frame
        """
        height, width, _ = image.shape
        ratio = height / width
        if self.use_sd_resolution:
            image = cv2.resize(image, (640, int(640 * ratio)))
        elif self.use_hd_resolution:
            image = cv2.resize(image, (1280, int(1280 * ratio)))
        elif self.use_fhd_resolution:
            image = cv2.resize(image, (1920, int(1920 * ratio)))
        else:
            pass
        return image

    def ocr(self, image):
        """
        Run OCR on the detected license plate
        """
        text = self.ocr_model(image)
        return text

    def run(self):
        """
        Run the pipeline end2end
        """
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FPS, 5)
        count = 0 # for counting detected plates
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                vehicle_results = self.vehicle_model(source=frame, save=False, conf=0.5, verbose=0)
                vehicle_boxes = vehicle_results[0].boxes.xyxy
                detected_plates = []
                for box in vehicle_boxes:
                    if box is None:
                        continue
                    box = box.cpu().numpy().astype(int)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), self.color["green"], thickness=2)
                    cropped_vehicle = frame[box[1]:box[3], box[0]:box[2], :] # crop a vehicle
                    plate_results = self.plate_model(source=cropped_vehicle, save=False, conf=0.1, verbose=0)
                    plate_boxes = plate_results[0].boxes.xyxy

                    # Displaying plate detection
                    for plate_box in plate_boxes:
                        plate_box = plate_box.cpu().numpy().astype(int)
                        src_point = (plate_box[0]+box[0], plate_box[1]+box[1])
                        dst_point = (plate_box[2]+box[0], plate_box[3]+box[1])
                        cv2.rectangle(frame, src_point, dst_point, self.color["red"], thickness=2)
                        cropped_plate = cropped_vehicle[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2], :]
                        detected_plates.append(cropped_plate)
                        if self.save_plate_image and check_image_size(cropped_plate, 64, 64):
                            detected_plates.append(cropped_plate)
                            filename = os.path.join(self.saved_path, str(count)+".jpg")
                            cv2.imwrite(filename, cropped_plate)
                            count += 1

                    # OCR the detected plate and display to monitor
                    if len(detected_plates) > 0:
                        cropped_vehicle = torch.from_numpy(cropped_vehicle)
                        ocr_text = self.ocr(cropped_vehicle)
                        pos = (box[0], box[1]-5)
                        draw_text(img = frame,
                                text = ocr_text,
                                pos = pos,
                                text_color=self.color["red"])

                # Display to monitor
                num_plate_info = "Detected plates: " + str(len(detected_plates))
                draw_text(img = frame,
                              text = num_plate_info,
                              pos = (0, 10),
                              font_scale=2,
                              text_color=self.color["red"])
                frame = self.set_resolution(frame)
                cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

if __name__ == "__main__":
    pipeline = Pipeline(source="data/TuKy.mp4",
                        vehicle_weight="weights/vehicle_yolov8.pt",
                        plate_weight="weights/plate_yolov8.pt",
                        use_sd_resolution=True,
                        save_plate_image=True)
    pipeline.run()