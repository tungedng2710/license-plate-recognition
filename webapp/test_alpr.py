import numpy as np
import cv2
import os
import argparse
from ultralytics import YOLO
from test_ocr import extract_plate_info
from utils.utils import BGR_COLORS, check_legit_plate, check_image_size, draw_text, \
    crop_expanded_plate


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_source", type=str, default="data/test_samples/xe4.jpg",
                        help="path to the image for testing")
    parser.add_argument("--vehicle_weight", type=str,
                        default="weights/vehicle_yolov8s_640.pt ",
                        help="path to the yolov8 weight of vehicle detector")
    parser.add_argument("--plate_weight", type=str,
                        default="weights/plate_yolov8n_320_2024.pt",
                        help="path to the yolov8 weight of plate detector")
    parser.add_argument("--vconf", type=float, default=0.6,
                        help="confidence for vehicle detection")
    parser.add_argument(
        "--pconf",
        type=float,
        default=0.25,
        help="confidence for plate detection")
    parser.add_argument(
        "--ocr_thres",
        type=float,
        default=0.8,
        help="threshold for ocr model")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="cuda id if available")
    return parser.parse_args()


class Vehicle:
    track_id: int = 0
    vehicle_detection_score: float = 0.0
    vehicle_type: str = ""
    vehicle_bbox: np.ndarray = None  # xyxy
    vehicle_image: np.ndarray = None
    license_plate_image: np.ndarray = None
    license_plate: str = ""
    license_plate_bbox: np.ndarray = None  # xyxy
    license_plate_score: float = 0.0
    tracking_feature_vector: np.ndarray
    is_recognized: bool = False


class ALPR:
    def __init__(self, opts) -> None:
        self.vehicle_detector = YOLO(opts.vehicle_weight, task='detect')
        self.plate_detector = YOLO(opts.plate_weight, task='detect')
        self.opts = opts
        self.vehicles = []
        self.vehicle_types = ['bus', 'car', 'motorcycle', 'truck', 'bicycle']
        self.color = BGR_COLORS

    def detect_vehicle(self, image):
        vehicle_detection = self.vehicle_detector(
            image,
            verbose=False,
            device=self.opts.device,
            imgsz=640,
            conf=self.opts.vconf)[0]
        vehicle_boxes = vehicle_detection.boxes
        vehicle_xyxy = vehicle_boxes.xyxy
        for idx, bbox in enumerate(vehicle_xyxy):
            bbox = bbox.cpu().numpy().astype(int)
            vehicle = Vehicle()
            vehicle.vehicle_bbox = bbox
            vehicle.vehicle_type = self.vehicle_types[int(
                vehicle_boxes.cls[idx])]
            vehicle.vehicle_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            self.vehicles.append(vehicle)

    def extract_license_plate(self):
        input_batch = [vehicle.vehicle_image for vehicle in self.vehicles]
        if len(input_batch) == 0:
            return
        plate_detections = self.plate_detector(
            input_batch,
            verbose=False,
            imgsz=320,
            device=self.opts.device,
            conf=self.opts.pconf)
        for idx, vehicle in enumerate(self.vehicles):
            plate_detection = plate_detections[idx]
            plate_boxes = plate_detection.boxes
            # if check_image_size(vehicle.vehicle_image, 112, 112):
            #     continue
            if len(plate_boxes.xyxy) > 0:
                plate_xyxy = plate_boxes.xyxy.cpu().numpy().astype(int)[0]
                vehicle.license_plate_bbox = plate_xyxy
                vehicle.plate_image = crop_expanded_plate(
                    plate_xyxy, vehicle.vehicle_image, 0.15)
                result = extract_plate_info(
                    vehicle.plate_image, self.opts.ocr_thres)
                vehicle.license_plate = result[0]
                vehicle.license_plate_score = result[1]

    def draw_result(self, ori_image):
        image = ori_image.copy()
        for vehicle in self.vehicles:
            vehicle_bbox = vehicle.vehicle_bbox
            src_point = (vehicle_bbox[0], vehicle_bbox[1])
            dst_point = (vehicle_bbox[2], vehicle_bbox[3])
            cv2.rectangle(image, src_point, dst_point, self.color['green'], 1)
            label = vehicle.vehicle_type
            if check_legit_plate(vehicle.license_plate):
                label += ' | ' + vehicle.license_plate
            draw_text(img=image,
                      text=label,
                      pos=src_point,
                      text_color=self.color["blue"],
                      text_color_bg=self.color["green"])
        return image

    def __call__(self, image):
        self.detect_vehicle(image)
        self.extract_license_plate()
        return self.draw_result(image)


def set_hd_resolution(image):
    """
    Set video resolution (for displaying only)
    Arg:
        image (OpenCV image): video frame read by cv2
    """
    height, width, _ = image.shape
    ratio = height / width
    image = cv2.resize(image, (1280, int(1280 * ratio)))
    return image


def input_source_is_video(opts):
    return opts.input_source.endswith(('.mp4', '.avi', '.mov'))


if __name__ == '__main__':
    opts = get_args()
    lp_recognizer = ALPR(opts)
    if not input_source_is_video(opts):
        image = cv2.imread(opts.input_source)
        result = lp_recognizer(image)
        cv2.imwrite("data/result.jpg", result)
    else:
        cap = cv2.VideoCapture(opts.input_source)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                lp_recognizer.vehicles = []
                image = frame.copy()
                image = lp_recognizer(image)
                cv2.imshow("Test ALPR", set_hd_resolution(image))
                del image

            key = cv2.waitKey(1)
            if key == ord('q'): # Quit video
                break
        cap.release()
        cv2.destroyAllWindows()
