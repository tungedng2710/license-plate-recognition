"""
Copyright (C) 2023 TonAI
"""
import argparse
import shutil
import os
import cv2
import torch

from ultralytics import YOLO
from tracking.utils.parser import get_config
from tracking.deep_sort import DeepSort
from utils.ocr import DummyOCR, EasyOCR
from utils.utils import map_label, check_image_size, draw_text, resize_, draw_box, \
    draw_tracked_boxes, compute_color, set_hd_resolution, BGR_COLORS, VEHICLES

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
    parser.add_argument("--config_deepsort", type=str, default="tracking/configs/deep_sort.yaml")
    return parser.parse_args()

class Pipeline():
    """
    License plate OCR pipeline
    Args:
    - source (str): path to video, 0 for webcam
    - vehicle_weight (str): path to the yolov8 weight of vehicle detector
    - plate_weight (str): path to the yolov8 weight of plate detector
    - save_result (bool): save cropped plate and output video to file
    """
    def __init__(self,
                 source: str = "data",
                 vehicle_weight: str = None,
                 plate_weight: str = None,
                 config_deepsort: str = None):
        # Core properties
        self.source = source
        self.vehicle_weight = vehicle_weight
        self.vehicle_model = YOLO(vehicle_weight)
        self.plate_model = YOLO(plate_weight)
        self.ocr_model = EasyOCR()

        # DeepSort
        self.config_deepsort = config_deepsort
        cfg = get_config()
        cfg.merge_from_file(self.config_deepsort)
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

        # Miscellaneous for displaying
        self.color = BGR_COLORS

    def ocr(self, image):
        """
        Run OCR on the detected license plate
        """
        text = self.ocr_model(image)
        return text

    def run(self,
            hd_resolution: bool = True,
            vconf: float = 0.6,
            pconf: float = 0.15,
            save_result: bool = False):
        """
        Run the pipeline end2end
        Args:
        - fancy_box (bool): draw fancy bounding box
        - vconf (float in [0,1]): confidence for vehicle detection
        - pconf (float in [0,1]): confidence for plate detection
        """
        cap = cv2.VideoCapture(self.source)

        # Config saved path
        if save_result:
            vid_name = os.path.basename(self.source).split('.')[0]
            saved_plate_path = f"data/{vid_name}"
            vid_name = os.path.basename(self.source)
            if os.path.exists(saved_plate_path):
                shutil.rmtree(saved_plate_path)
            os.makedirs(saved_plate_path)
        
        # -------------------------- MAIN --------------------------
        count = 0 # for counting total detected plates from whole video
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is not None:
                displayed_frame = frame.copy()
            else:
                break
            if ret:
                # Detect vehicles
                vehicle_results = self.vehicle_model(source=frame, conf=vconf, verbose=False)
                vehicle_boxes = vehicle_results[0].boxes.xyxy # Bounding box
                vehicle_labels = vehicle_results[0].boxes.cls # Predicted classes
                detected_plates = []

                # ---------------------- Tracking ----------------------
                xywhs = vehicle_results[0].boxes.xywh.cpu()
                confss = vehicle_results[0].boxes.conf.cpu()
                outputs = self.deepsort.update(xywhs, confss, frame)
                # Draw tracked boxes
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_tracked_boxes(displayed_frame, bbox_xyxy, identities)
                # ------------------------------------------------------

                for index, box in enumerate(vehicle_boxes):
                    label_name = map_label(int(vehicle_labels[index]), VEHICLES)
                    have_plate = False
                    if box is None:
                        continue
                    box = box.cpu().numpy().astype(int)
                    # Draw object information
                    draw_text(img = displayed_frame, text = label_name, 
                              pos = (int((box[0]+box[2])/2), box[1]),
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
                        cv2.rectangle(displayed_frame, src_point, dst_point, self.color["red"], thickness=2)
                        cropped_plate = cropped_vehicle[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2], :]

                        # Zoom in and display the plate
                        displayed_plate = resize_(cropped_plate, 3)
                        plate_pos = (int(displayed_plate.shape[0]) + box[1],
                                     int(displayed_plate.shape[1]) + box[0])
                        adjust_height = int((box[3]-box[1]) / 2)
                        try:
                            displayed_frame[box[1]+adjust_height:plate_pos[0]+adjust_height, \
                                            box[0]:plate_pos[1], :] = displayed_plate
                        except:
                            pass

                        if check_image_size(cropped_plate, 8, 8): # Ignore plates smaller than 8x8
                            have_plate = True
                            detected_plates.append(cropped_plate)
                            # Save the cropped plate, ignore if its size smaller than 32x32
                            if save_result and check_image_size(cropped_plate, 32, 32):
                                filename = os.path.join(saved_plate_path, str(count)+".jpg")
                                cv2.imwrite(filename, cropped_plate)
                                count += 1

                        # OCR the detected plate and display to monitor
                        if have_plate:
                            cropped_vehicle = torch.from_numpy(cropped_vehicle)
                            # -------------- OCR module --------------
                            ocr_text = self.ocr(cropped_plate)
                            # ----------------------------------------
                            # Display to monitor
                            pos = (int((box[0]+box[2])/2), box[1])
                            info = f"{label_name} {ocr_text}"
                            draw_text(img = displayed_frame, text = info, pos = pos,
                                      text_color=self.color["blue"],
                                      text_color_bg=self.color["green"])

                # Display global detection info
                num_plate_info = "Detected plates: " + str(len(detected_plates))
                draw_text(img = displayed_frame, text = num_plate_info, pos = (0, 0), font_scale=1,
                          text_color=self.color["black"],
                          text_color_bg=self.color["amber"])
                if save_result:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps, w, h = 30, displayed_frame.shape[1], displayed_frame.shape[0]
                    vid_writer = cv2.VideoWriter(f"data/infered_{vid_name}", fourcc, fps, (w, h))
                    vid_writer.write(displayed_frame)
                if hd_resolution:
                    displayed_frame = set_hd_resolution(displayed_frame)
                cv2.imshow("TonVision", displayed_frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        cap.release()
        if save_result:
            vid_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    args = get_args()
    pipeline = Pipeline(source=args.source,
                        vehicle_weight=args.vehicle_weight,
                        plate_weight=args.plate_weight,
                        config_deepsort=args.config_deepsort)
    pipeline.run(vconf=args.vconf,
                 pconf=args.pconf,
                 hd_resolution = True,
                 save_result=args.save)
