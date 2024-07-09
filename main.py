"""
Copyright (C) 2023 TonAI
"""
import argparse
import os
import cv2
import numpy as np
import torch
import re
from time import time

from ultralytics import YOLO
from tracking.deep_sort import DeepSort
from tracking.sort import Sort
from utils.utils import map_label, check_image_size, draw_text, check_legit_plate, \
    gettime, compute_color, argmax, BGR_COLORS, VEHICLES, crop_expanded_plate
from ppocr_onnx import DetAndRecONNXPipeline as PlateReader
# from ultralytics.utils.checks import check_requirements


def get_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        default="test_urban.mp4",
        help="path to video, 0 for webcam")
    parser.add_argument("--vehicle_weight", type=str,
                        default="weights/vehicle_yolov8n_1088_v2.pt",
                        help="path to the yolov8 weight of vehicle detector")
    parser.add_argument("--plate_weight", type=str,
                        default="weights/plate_yolov8n_9k.pt",
                        help="path to the yolov8 weight of plate detector")
    parser.add_argument("--dsort_weight", type=str,
                        default="weights/deepsort/deepsort.onnx",
                        help="path to the weight of DeepSORT tracker")
    parser.add_argument(
        "--ocr_weight",
        type=str,
        default="weights/plate_ppocr/",
        help="path to the paddle ocr weight of plate recognizer")
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
        default=0.95,
        help="threshold for ocr model")
    parser.add_argument(
        "--deepsort",
        action="store_true",
        help="suse DeepSORT tracking instead of normal SORT")
    parser.add_argument(
        "--read_plate",
        action="store_true",
        help="read plate information")
    parser.add_argument(
        "--save",
        action="store_true",
        help="save output video")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="real-time monitoring")
    parser.add_argument(
        "--show_plate",
        action="store_true",
        help="zoom in detected plate")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/logs",
        help="saved path")
    parser.add_argument(
        "--lang",
        type=str,
        default="vi",
        help="language to show (vi, en, es, fr)")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="cuda id if available")
    return parser.parse_args()


class TrafficCam():
    """
    License plate OCR TrafficCam
    Args:
    - opts: parsed arguments
    """

    def __init__(self, opts):
        self.opts = opts
        # Core properties
        self.video = opts.video
        self.vehicle_detector = YOLO(opts.vehicle_weight, task='detect')
        self.plate_detector = YOLO(opts.plate_weight, task='detect')
        self.read_plate = opts.read_plate
        if self.read_plate:
            self.plate_reader = PlateReader(
                text_det_onnx_model="weights/ppocrv4/ch_PP-OCRv4_det_infer.onnx",
                text_rec_onnx_model="weights/ppocrv4/ch_PP-OCRv4_rec_infer.onnx",
                box_thresh=0.6)
        self.ocr_thres = opts.ocr_thres

        # DeepSort Tracking
        self.deepsort = opts.deepsort
        self.dsort_weight = opts.dsort_weight
        self.init_tracker()

        # Miscellaneous for displaying
        self.color = BGR_COLORS
        self.show_plate = opts.show_plate
        self.stream = opts.stream
        self.lang = opts.lang
        self.save_dir = opts.save_dir
        self.save = opts.save

    def extract_plate(self, plate_image):
        results = self.plate_reader.detect_and_ocr(plate_image)
        if len(results) > 0:
            plate_info = ''
            conf = []
            for result in results:
                plate_info += result.text + ' '
                conf.append(result.score)
            conf = sum(conf) / len(conf)
            return re.sub(r'[^A-Za-z0-9\-.]', '', plate_info), conf
        else:
            return '', 0

    def init_tracker(self):
        """
        Initialize tracker
        """
        if self.deepsort:
            print("Using DeepSORT Tracking")
            self.tracker = DeepSort(self.dsort_weight, max_dist=0.2,
                                    min_confidence=0.3, nms_max_overlap=0.5,
                                    max_iou_distance=0.7, max_age=70,
                                    n_init=3, nn_budget=100,
                                    use_cuda=torch.cuda.is_available())
        else:
            self.tracker = Sort()
        self.vehicles_dict = {}

    def run(self):
        """
        Run the TrafficCam end2end
        """
        # Config video properties
        vid_name = os.path.basename(self.video)
        cap = cv2.VideoCapture(self.video)
        title = "Traffic Surveillance"
        if self.stream:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.save:
            log_path = self.save_dir
            frames_path = os.path.join(log_path, "frames")
            detected_objects_path = os.path.join(log_path, "objects")
            detected_plates_path = os.path.join(log_path, "plates")
            if not os.path.exists(log_path):
                os.makedirs(log_path)
                os.makedirs(frames_path)
                os.makedirs(detected_objects_path)
                os.makedirs(detected_plates_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vid_writer = cv2.VideoWriter(
                f"{log_path}/infered_{vid_name}", fourcc, fps, (w, h))
        num_frame = 0
        captured = 0
        thresh_h = int(h / 5)  # Limit detection zone
        print("Traffic Cam is ready!")
        while cap.isOpened():
            t0_fps = gettime()
            ret, frame = cap.read()
            num_frame += 1
            if int(num_frame) == 500:
                self.init_tracker()
            # frame = cv2.imread("data/test_samples/images/bienso/biendo15.jpg")
            if frame is not None:
                displayed_frame = frame.copy()
            else:
                continue
            if ret:
                # cv2.line(displayed_frame, (0, thresh_h), (w, thresh_h), self.color["blue"], 2)
                """
                --------------- VEHICLE DETECTION ---------------
                Plate recognition include two subsections: detection and tracking
                    - Detection: Ultralytics YOLOv8
                    - Tracking: DeepSORT
                """
                t1 = time()
                vehicle_detection = self.vehicle_detector(
                    frame,
                    verbose=False,
                    imgsz=640,
                    device=self.opts.device,
                    conf=self.opts.vconf)[0]
                # print(f"Inference time: {time() - t1}")
                vehicle_boxes = vehicle_detection.boxes
                vehicle_xyxy = vehicle_boxes.xyxy
                vehicle_labels = vehicle_boxes.cls
                try:
                    if self.deepsort:
                        outputs = self.tracker.update(vehicle_boxes.cpu().xywh,
                                                      vehicle_boxes.cpu().conf,
                                                      frame)
                    else:
                        outputs = self.tracker.update(
                            vehicle_boxes.cpu().xyxy).astype(int)
                except BaseException:
                    continue
                in_frame_indentities = []

                for idx in range(len(outputs)):
                    identity = outputs[idx, -1]
                    in_frame_indentities.append(identity)
                    if str(identity) not in self.vehicles_dict:
                        self.vehicles_dict[str(identity)] = {"save": False,
                                                             "saved_plate": False,
                                                             "plate_image": None,
                                                             "vehicle_image": None}
                    self.vehicles_dict[str(
                        identity)]["bbox_xyxy"] = outputs[idx, :4]
                    vehicle_bbox = self.vehicles_dict[str(
                        identity)]["bbox_xyxy"]
                    src_point = (vehicle_bbox[0], vehicle_bbox[1])
                    dst_point = (vehicle_bbox[2], vehicle_bbox[3])
                    color = compute_color(identity)
                    cv2.rectangle(
                        displayed_frame, src_point, dst_point, color, 1)

                # for index, box in enumerate(vehicle_xyxy):
                #     if box is None:
                #         continue
                #     label_name = map_label(int(vehicle_labels[index]), VEHICLES[self.lang])
                #     box = box.cpu().numpy().astype(int)
                #     draw_text(img=displayed_frame, text=label_name,
                #               pos=(box[0], box[1]),
                #               text_color=self.color["blue"],
                #               text_color_bg=self.color["green"])

                """
                --------------- PLATE RECOGNITION ---------------
                This section will run if --read-plate
                Plate recognition include two subsections: detection and OCR
                    - Detection: Ultralytics YOLOv8
                    - Optical Character Recognition: Baidu PaddleOCR
                """
                if self.read_plate:
                    active_vehicles = []
                    input_batch = []
                    for identity in in_frame_indentities:
                        vehicle = self.vehicles_dict[str(identity)]
                        if "ocr_conf" not in vehicle:
                            vehicle["ocr_conf"] = 0.0
                            vehicle["plate_number"] = "nan"
                        box = vehicle["bbox_xyxy"].astype(int)
                        plate_number = self.vehicles_dict[str(
                            identity)]["plate_number"]
                        success = (vehicle["ocr_conf"] > self.ocr_thres) \
                            and len(plate_number) > 5 \
                            and check_legit_plate(plate_number)
                        if success:
                            pos = (box[0], box[1] + 26)
                            draw_text(
                                img=displayed_frame,
                                text=plate_number,
                                pos=pos,
                                text_color=self.color["blue"],
                                text_color_bg=self.color["green"])
                            if self.save and not vehicle["save"]:
                                cropped_vehicle = frame[box[1]
                                    :box[3], box[0]:box[2], :]
                                # cv2.imwrite(f"{detected_objects_path}/{plate_number}.jpg", cropped_vehicle)
                                if check_image_size(vehicle["plate_image"], 32, 16):
                                    cv2.imwrite(
                                        f"{detected_plates_path}/{plate_number}.jpg",
                                        vehicle["plate_image"])
                                    if cropped_vehicle is not None:
                                        cv2.imwrite(
                                            f"{detected_objects_path}/{plate_number}.jpg", cropped_vehicle)
                                    del vehicle["plate_image"]
                                    del vehicle["vehicle_image"]
                                    vehicle["vehicle_image"] = None
                                    vehicle["plate_image"] = None
                                    vehicle["save"] = True
                            continue
                        else:
                            # if box[1] < thresh_h: # Ignore vehicle out of recognition zone
                            #     in_frame_indentities.remove(identity)
                            #     continue
                            # crop vehicle image to push into the plate
                            # detector
                            cropped_vehicle = frame[box[1]
                                :box[3], box[0]:box[2], :]
                            vehicle["vehicle_image"] = cropped_vehicle
                            if not check_image_size(
                                    cropped_vehicle, 112, 112):  # ignore too small image!
                                continue
                            input_batch.append(cropped_vehicle)
                            active_vehicles.append(vehicle)
                    if len(input_batch) > 0:
                        plate_detections = self.plate_detector(
                            input_batch,
                            verbose=False,
                            imgsz=320,
                            device=self.opts.device,
                            conf=self.opts.pconf)
                        vehicle_having_plate = []
                        for id, detection in enumerate(plate_detections):
                            vehicle = active_vehicles[id]
                            cropped_vehicle = input_batch[id]
                            box = vehicle["bbox_xyxy"].astype(int)
                            plate_xyxy = detection.boxes.xyxy
                            if len(plate_xyxy) < 1:
                                continue
                            # Display plate detection
                            plate_xyxy = plate_xyxy[0]
                            plate_xyxy = plate_xyxy.cpu().numpy().astype(int)
                            src_point = (
                                plate_xyxy[0] + box[0], plate_xyxy[1] + box[1])
                            dst_point = (
                                plate_xyxy[2] + box[0], plate_xyxy[3] + box[1])
                            cv2.rectangle(
                                displayed_frame,
                                src_point,
                                dst_point,
                                self.color["green"],
                                thickness=2)
                            # cropped_plate = cropped_vehicle[plate_xyxy[1]:plate_xyxy[3], \
                            # plate_xyxy[0]:plate_xyxy[2], :]
                            try:
                                cropped_plate = crop_expanded_plate(
                                    plate_xyxy, cropped_vehicle, 0.15)
                            except BaseException:
                                cropped_plate = np.zeros((8, 8, 3))
                            vehicle["plate_image"] = cropped_plate
                            vehicle_having_plate.append(vehicle)

                        if len(vehicle_having_plate) > 0:
                            for vehicle in vehicle_having_plate:
                                plate_info, conf = self.extract_plate(
                                    vehicle["plate_image"])
                                cur_ocr_conf = vehicle["ocr_conf"]
                                if conf > cur_ocr_conf:
                                    vehicle["plate_number"] = plate_info
                                    vehicle["ocr_conf"] = conf

                #  ---------------- MISCELLANEOUS ---------------- #
                # ids = list(map(int, list(self.vehicles_dict.keys())))
                # num_vehicle = 0 if len(ids) == 0 else max(ids)
                t = gettime() - t0_fps
                fps_info = int(round(1 / t, 0))
                global_info = f"FPS: {fps_info}"
                draw_text(img=displayed_frame, text=global_info,
                          font_scale=1, font_thickness=2,
                          text_color=self.color["blue"],
                          text_color_bg=self.color["white"])
                if self.save:  # Save inference result to file
                    vid_writer.write(displayed_frame)
                    if int(num_frame) == int(
                            fps * 10):  # save frame every 10 seconds
                        self.init_tracker()
                        cv2.imwrite(
                            f"{frames_path}/{captured}.jpg",
                            displayed_frame)
                        captured += 1
                if self.stream:
                    cv2.imshow(title, displayed_frame)
                key = cv2.waitKey(1)
                if key == ord('q'):  # Quit video
                    break
                if key == ord('r'):  # Reset tracking
                    self.init_tracker()
                if key == ord('p'):  # Pause video
                    cv2.waitKey(-1)
                del displayed_frame
                del frame
        cap.release()
        if self.save:
            vid_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    opts = get_args()
    TrafficCam = TrafficCam(opts)
    TrafficCam.run()
