"""
Copyright (C) 2023 TonAI
"""
import argparse
import shutil
import os
import cv2
import torch
import datetime

from ultralytics import YOLO
from tracking.utils.parser import get_config
from tracking.deep_sort import DeepSort
from utils.ocr import PPOCR
from utils.utils import map_label, check_image_size, draw_text, resize_, draw_box, \
    draw_tracked_boxes, compute_color, set_hd_resolution, BGR_COLORS, VEHICLES


def get_args():
    """
    Get parsing arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/cam1.mp4", help="path to video, 0 for webcam")
    parser.add_argument("--vehicle_weight", type=str,
                        default="weights/vehicle_yolov8s.pt",
                        help="path to the yolov8 weight of vehicle detector")
    parser.add_argument("--plate_weight", type=str,
                        default="weights/plate_yolov8n.pt",
                        help="path to the yolov8 weight of plate detector")
    parser.add_argument("--vconf", type=float, default=0.6, help="confidence for vehicle detection")
    parser.add_argument("--pconf", type=float, default=0.25, help="confidence for plate detection")
    parser.add_argument("--save", action="store_true", help="save cropped detected objects")
    parser.add_argument("--config_deepsort", type=str, default="tracking/configs/deep_sort.yaml")
    return parser.parse_args()


class Pipeline():
    """
    License plate OCR pipeline
    Args:
    - video (str): path to video, 0 for webcam
    - vehicle_weight (str): path to the yolov8 weight of vehicle detector
    - plate_weight (str): path to the yolov8 weight of plate detector
    - save_result (bool): save cropped plate and output video to file
    """

    def __init__(self,
                 video: str = "data",
                 vehicle_weight: str = None,
                 plate_weight: str = None,
                 config_deepsort: str = None):
        # Core properties
        self.video = video
        self.vehicle_weight = vehicle_weight
        self.vehicle_model = YOLO(vehicle_weight)
        self.plate_model = YOLO(plate_weight)
        self.ocr_model = PPOCR()

        # DeepSort
        self.config_deepsort = config_deepsort
        cfg = get_config()
        cfg.merge_from_file(self.config_deepsort)
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                 max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                                 nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)

        # Miscellaneous for displaying
        self.color = BGR_COLORS

    def ocr(self, image):
        """
        Run OCR on the detected license plate
        """
        text, conf = self.ocr_model(image)
        return text, conf

    def run(self,
            hd_resolution: bool = True,
            vconf: float = 0.6,
            pconf: float = 0.15,
            ocrconf_thres: float = 0.7,
            save_result: bool = False,
            stream: bool = False
            ):
        """
        Run the pipeline end2end
        Args:
        - fancy_box (bool): draw fancy bounding box
        - vconf (float in [0,1]): confidence for vehicle detection
        - pconf (float in [0,1]): confidence for plate detection
        - ocrconf (float in [0,1]): confidence for OCR
        - save_result (bool): save the result to file
        - stream (bool): show real-time video stream
        """
        # Get video properties
        vid_name = os.path.basename(self.video)
        cap = cv2.VideoCapture(self.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Config logs
        if save_result:
            log_path = f"data/logs"
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vid_writer = cv2.VideoWriter(f"{log_path}/infered_{vid_name}", fourcc, fps, (w, h))

        # -------------------------- MAIN --------------------------
        count = 0  # Count total detected plates from whole video
        # thresh_h = int(h / 3) # Limit detection zone
        
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is not None:
                displayed_frame = frame.copy()
            else:
                break
            if ret:
                # # Draw a line to separate detection zone
                # cv2.line(displayed_frame, (0, thresh_h), (w, thresh_h), self.color["blue"], 1)
                # Detect vehicles
                vehicle_results = self.vehicle_model(source=frame, conf=vconf, verbose=False)
                vehicle_boxes = vehicle_results[0].boxes.xyxy  # Bounding box
                vehicle_labels = vehicle_results[0].boxes.cls  # Predicted classes
                detected_plates = []

                # ---------------------- Tracking ----------------------
                xywhs = vehicle_results[0].boxes.xywh.cpu()
                confss = vehicle_results[0].boxes.conf.cpu()
                outputs = self.deepsort.update(xywhs, confss, frame)
                # Draw tracked boxes
                vehicles = []
                for idx in range(len(outputs)):
                    vehicle = {
                        "id": outputs[idx, -1],
                        "bbox_xyxy": outputs[idx, :4],
                        "plate_number": "unavailable"
                    }
                    vehicles.append(vehicle)
                    vehicle_bbox = vehicle["bbox_xyxy"]
                    src_point = (vehicle_bbox[0], vehicle_bbox[1])
                    dst_point = (vehicle_bbox[2], vehicle_bbox[3])
                    cv2.rectangle(displayed_frame, src_point, dst_point, self.color["green"], thickness=2)
                # ------------------------------------------------------
                for index, box in enumerate(vehicle_boxes):
                    label_name = map_label(int(vehicle_labels[index]), VEHICLES)
                    have_plate = False
                    if box is None:
                        continue
                    box = box.cpu().numpy().astype(int)
                    # Draw object information
                    draw_text(img=displayed_frame, text=label_name,
                              pos=(int((box[0] + box[2]) / 2), box[1]),
                              text_color=self.color["blue"],
                              text_color_bg=self.color["green"])
                ocr_conf = 0.0
                for vehicle in vehicles:
                    box = vehicle["bbox_xyxy"].astype(int)
                    # # Adjust the box to focus to the potential region
                    # focused_box = [box[0], int((box[3] + box[1]) / 2), box[2], int(box[3] * 1.2)]
                    focused_box = box
                    cropped_vehicle = frame[focused_box[1]:focused_box[3], focused_box[0]:focused_box[2], :]
                    plate_results = self.plate_model(source=cropped_vehicle, conf=pconf, verbose=False)
                    plate_boxes = plate_results[0].boxes.xyxy

                    # Displaying plate detection
                    for plate_box in plate_boxes:
                        plate_box = plate_box.cpu().numpy().astype(int)
                        src_point = (plate_box[0] + focused_box[0], plate_box[1] + focused_box[1])
                        dst_point = (plate_box[2] + focused_box[0], plate_box[3] + focused_box[1])
                        # if src_point[1] < thresh_h:
                        #     continue
                        cv2.rectangle(displayed_frame, src_point, dst_point, self.color["red"], thickness=2)
                        cropped_plate = cropped_vehicle[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2], :]
                        if check_image_size(cropped_plate, 16, 16):   # Ignore small plates
                            have_plate = True
                            # Zoom in and display the plate
                            displayed_plate = resize_(cropped_plate, 3)
                            plate_pos = (int(displayed_plate.shape[0]) + box[1],
                                         int(displayed_plate.shape[1]) + box[0])
                            adjust_height = int((box[3] - box[1]) / 2)
                            try:
                                displayed_frame[box[1] + adjust_height:plate_pos[0] + adjust_height, \
                                box[0]:plate_pos[1], :] = displayed_plate
                            except:
                                pass
                            detected_plates.append(cropped_plate)

                        # OCR the detected plate and display to monitor
                        if have_plate:
                            # -------------- OCR module --------------
                            ocr_text, ocr_conf = self.ocr(cropped_plate)
                            vehicle["ocr_conf"] = ocr_conf
                            if vehicle["ocr_conf"] >= ocrconf_thres:
                                vehicle["plate_number"] = ocr_text
                                # Display to monitor
                                pos = (int((box[0] + box[2]) / 2), box[1])
                                plate_number = vehicle["plate_number"]
                                info = f"{label_name} {plate_number}"
                                draw_text(img=displayed_frame, text=info, pos=pos,
                                          text_color=self.color["blue"],
                                          text_color_bg=self.color["green"])
                            else:
                                vehicle["plate_number"] = "unavailable"

                if save_result:
                    vid_writer.write(displayed_frame)
                if hd_resolution:
                    displayed_frame = set_hd_resolution(displayed_frame)
                if stream:
                    cv2.imshow("TonTraffic", displayed_frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        cap.release()
        if save_result:
            vid_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    opts = get_args()
    root_dir = "data/traffic_video_clean"
    for video_name in os.listdir(root_dir):
        if "mp4" in video_name:
            print("Run inference on", os.path.join(root_dir, video_name))
            pipeline = Pipeline(video=os.path.join(root_dir, video_name),
                                vehicle_weight=opts.vehicle_weight,
                                plate_weight=opts.plate_weight,
                                config_deepsort=opts.config_deepsort) 
            try:
                pipeline.run(vconf=opts.vconf,
                            pconf=opts.pconf,
                            ocrconf_thres=0.9,
                            hd_resolution=True,
                            save_result=True,
                            stream=False)
            except:
                continue
    # pipeline = Pipeline(video=opts.video,
    #                     vehicle_weight=opts.vehicle_weight,
    #                     plate_weight=opts.plate_weight,
    #                     config_deepsort=opts.config_deepsort)
    # pipeline.run(vconf=opts.vconf,
    #              pconf=opts.pconf,
    #              hd_resolution = True,
    #              save_result=opts.save)
