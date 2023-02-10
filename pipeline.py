"""
Copyright (C) 2023 TonAI
"""
import argparse
import shutil
import os
import numpy as np
import cv2
import torch

from ultralytics import YOLO

from utils.ocr import DummyOCR, EasyOCR, VietOCR
from utils.utils import map_label, check_image_size, draw_text, draw_box, \
    BGR_COLORS, VEHICLES

def delete_file(path):
    """
    Delete generated file during inference
    """
    if os.path.exists(path):
        os.remove(path)

def get_args():
    """
    Get parsing arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/cam1.mp4", help="path to video, 0 for webcam")
    parser.add_argument("--vehicle_weight", type=str,
                        default="weights/vehicle_yolov8n.pt",
                        help="path to the yolov8 weight of vehicle detector")
    parser.add_argument("--plate_weight", type=str,
                        default="weights/plate_yolov8n.pt",
                        help="path to the yolov8 weight of plate detector")
    parser.add_argument("--vconf", type=float, default=0.6, help="confidence for vehicle detection")
    parser.add_argument("--pconf", type=float, default=0.15, help="confidence for plate detection")
    parser.add_argument("--save", action="store_true", help="save cropped detected objects")
    return parser.parse_args()

class Pipeline():
    """
    License plate OCR pipeline
    Args:
    - source (str): path to video, 0 for webcam
    - vehicle_weight (str): path to the yolov8 weight of vehicle detector
    - plate_weight (str): path to the yolov8 weight of plate detector
    - save_plate_image (bool): save cropped object to file
    """
    def __init__(self,
                 source: str = "data",
                 vehicle_weight: str = None,
                 plate_weight: str = None,
                 use_sd_resolution: bool = False,
                 use_hd_resolution: bool = False,
                 use_fhd_resolution: bool = False,
                 save_plate_image: bool = False):
        # Core properties
        self.source = source
        self.vehicle_weight = vehicle_weight
        self.vehicle_model = YOLO(vehicle_weight)
        self.plate_model = YOLO(plate_weight)
        self.ocr_model = EasyOCR()

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
            self.saved_path = f"data/{vid_name}"
            if os.path.exists(self.saved_path):
                shutil.rmtree(self.saved_path)
            os.makedirs(self.saved_path)

    def set_resolution(self, image):
        """
        Set video resolution (for displaying only)
        Arg:
            image (OpenCV image): video frame read by cv2
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

    def run(self,
            vconf: float = 0.6,
            pconf: float = 0.15):
        """
        Run the pipeline end2end
        Args:
        - vconf (float in [0,1]): confidence for vehicle detection
        - pconf (float in [0,1]): confidence for plate detection
        """
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FPS, 30)
        count = 0 # for counting total detected plates from whole video
        plate_batch = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Detect vehicles
                vehicle_results = self.vehicle_model(source=frame, conf=vconf, verbose=False)
                vehicle_boxes = vehicle_results[0].boxes.xyxy # Bounding box
                vehicle_labels = vehicle_results[0].boxes.cls # Predicted classes
                detected_plates = []
                for index, box in enumerate(vehicle_boxes):
                    label_name = map_label(int(vehicle_labels[index]), VEHICLES)
                    have_plate = False
                    if box is None:
                        continue
                    box = box.cpu().numpy().astype(int)
                    draw_box(frame, (box[0], box[1]), (box[2], box[3]), self.color["green"], 1, 0, 25)
                    draw_text(img = frame, text = label_name, pos = (box[0], box[1]),
                              text_color=self.color["blue"],
                              text_color_bg=self.color["green"])

                    # Adjust the box to focus to the potential region
                    focused_box = [box[0], int((box[3] + box[1]) / 2), box[2], int(box[3] * 1.2)]
                    cropped_vehicle = frame[focused_box[1]:focused_box[3], focused_box[0]:focused_box[2], :]
                    plate_results = self.plate_model(source=cropped_vehicle, conf=pconf, verbose=False)
                    plate_boxes = plate_results[0].boxes.xyxy

                    # Displaying plate detection
                    for plate_box in plate_boxes:
                        plate_box = plate_box.cpu().numpy().astype(int)
                        src_point = (plate_box[0]+focused_box[0], plate_box[1]+focused_box[1])
                        dst_point = (plate_box[2]+focused_box[0], plate_box[3]+focused_box[1])
                        cv2.rectangle(frame, src_point, dst_point, self.color["red"], thickness=2)
                        cropped_plate = cropped_vehicle[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2], :]
                        if check_image_size(cropped_plate, 8, 8): # Ignore plates smaller than 8x8
                            have_plate = True
                            detected_plates.append(cropped_plate)
                            # Save the cropped plate, ignore if its size smaller than 32x32
                            if self.save_plate_image and check_image_size(cropped_plate, 32, 32):
                                filename = os.path.join(self.saved_path, str(count)+".jpg")
                                cv2.imwrite(filename, cropped_plate)
                                plate_batch.append(cropped_plate)
                                count += 1

                        # OCR the detected plate and display to monitor
                        if have_plate:
                            cropped_vehicle = torch.from_numpy(cropped_vehicle)
                            # OCR module
                            ocr_text = self.ocr(cropped_plate)
                            # Display to monitor
                            pos = (box[0], box[1])
                            info = f"{label_name} {ocr_text}"
                            draw_text(img = frame, text = info, pos = pos,
                                    text_color=self.color["blue"],
                                    text_color_bg=self.color["green"])

                # Display detection info to monitor
                num_plate_info = "Detected plates: " + str(len(detected_plates))
                draw_text(img = frame, text = num_plate_info, pos = (0, 0), font_scale=2,
                          text_color=self.color["black"],
                          text_color_bg=self.color["amber"])
                frame = self.set_resolution(frame)
                cv2.imshow("TonVision", frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    delete_file("temp.jpg")
                    break

            # Create a batch of detections
            if len(plate_batch) == 32:
                plate_batch = np.array(plate_batch)
                plate_batch = torch.from_numpy(plate_batch)
                plate_batch = []

if __name__ == "__main__":
    args = get_args()
    pipeline = Pipeline(source=args.source,
                        vehicle_weight=args.vehicle_weight,
                        plate_weight=args.plate_weight,
                        use_hd_resolution=True,
                        save_plate_image=args.save)
    pipeline.run(vconf=args.vconf, pconf=args.pconf)
