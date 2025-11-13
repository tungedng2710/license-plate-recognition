"""
There are utilities, some of them are useful, but the other ones useless ._.
I'm too lazy to get rid of useless functions
"""
import re
import os
import cv2
import time
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

BGR_COLORS = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "amber": (0, 191, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0)
}
VEHICLES = {
    'vi': ['xe dap', 'xe buyt', 'o to', 'xe may', 'xe tai'],
    'en': ['bicycle', 'bus', 'car', 'motorbike', 'truck'],
    'es': ['bicicleta', 'autobus', 'auto', 'moto', 'camion'],
    'fr': ['velo', 'bus', 'voiture', 'moto', 'camion'],
    'coco': ['bicycle', 'bus', 'car', 'motorcycle', 'truck'],
    'coco_vi': ['xe dap', 'xe buyt', 'o to', 'xe may', 'xe tai']
}
COLOR_PALETTE = np.random.uniform(0, 255, size=(len(VEHICLES['en']), 3))

COLOR_HSV = {
        'black': [(0, 0, 0), (180, 255, 30)],
        'blue': [(100, 150, 0), (140, 255, 255)],
        'brown': [(10, 100, 20), (20, 255, 200)],  # Brown color range
        'gray': [(0, 0, 50), (180, 50, 200)],      # Lighter gray color range
        'orange': [(10, 100, 20), (25, 255, 255)], # Orange color range
        'pink': [(140, 50, 50), (170, 255, 255)],  # Pink color range
        'purple': [(125, 50, 50), (155, 255, 255)],# Purple color range
        'red': [(0, 70, 50), (10, 255, 255)],      # Red color range
        'white': [(0, 0, 200), (180, 20, 255)],    # White color range
        'yellow': [(20, 100, 100), (30, 255, 255)] # Yellow color range
    }

class MyDict(dict):
    def __getattribute__(self, item):
        return self[item]


@dataclass
class Vehicle:
    """Container for vehicle tracking and recognition metadata."""

    track_id: int
    vehicle_type: str = ""  # Detected vehicle category
    bbox_xyxy: Optional[np.ndarray] = None  # Vehicle bounding box [x1, y1, x2, y2]
    vehicle_image: Optional[np.ndarray] = None  # Cropped vehicle image
    plate_image: Optional[np.ndarray] = None  # Cropped license plate image
    plate_number: str = "nan"  # Recognized plate text
    ocr_conf: float = 0.0  # Confidence of OCR result
    license_plate_bbox: Optional[np.ndarray] = None  # Plate bbox in image coords
    vehicle_detection_score: float = 0.0
    license_plate_score: float = 0.0
    tracking_feature_vector: Optional[np.ndarray] = None
    save: bool = False  # Whether vehicle/plate images have been saved
    saved_plate: bool = False
    is_recognized: bool = False

def draw_detections(img, box, class_id, lang='en'):
    """
    Draws bounding boxes and labels on the input image based on the detected objects.

    Args:
        img: The input image to draw detections on.
        box: Detected bounding box.
        score: Corresponding detection score.
        class_id: Class ID for the detected object.

    Returns:
        None
    """
    x1, y1, w, h = box[0], box[1], box[2], box[3]
    color = COLOR_PALETTE[class_id]
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
    label = f'{map_label(class_id, VEHICLES[lang])}'
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                    cv2.FILLED)
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img

def gettime():
    return time.time()


def map_label(class_idx, vehicle_labels):
    """
    Map argmax output to label name following COCO Object
    """
    return vehicle_labels[class_idx]


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

def strip_extra_text(s: str) -> str:
    if len(s) > 6:
        if s[0].isalpha():
            for i, ch in enumerate(s):
                if ch.isdigit():
                    s = s[i:]
                    break
        if s and s[-1].isalpha():
            for j in range(len(s)-1, -1, -1):
                if s[j].isdigit():
                    s = s[:j+1]
                    break
    return s

def draw_text(img, text,
              pos=(0, 0),
              font=cv2.FONT_HERSHEY_SIMPLEX,
              font_scale=1,
              font_thickness=2,
              text_color=(255, 255, 255),
              text_color_bg=(0, 0, 0)):
    """
    Minor modification of cv2.putText to add background color
    """
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    bg_h = int(text_h * 1.05)  # expand the background height a bit
    cv2.rectangle(img, pos, (x + text_w, y + bg_h), text_color_bg, -1)
    pos = (x, y + text_h + font_scale)
    cv2.putText(img, text, pos, font, font_scale, text_color, font_thickness, cv2.LINE_AA)


def draw_box(img, pt1, pt2, color, thickness, r, d):
    """
    Draw more fancy bounding box
    """
    x1, y1 = pt1
    x2, y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def draw_tracked_boxes(img, bbox, identities=None, offset=(0, 0)):
    """
    Draw box tracked by deepsort
    """
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        idx = int(identities[i]) if identities is not None else 0
        color = compute_color(idx)
        label = '{}{:d}'.format("", idx)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def compute_color(label):
    """
    Add borders of different colors
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


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


def resize_(image, scale):
    """
    Compress cv2 resize function
    """
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dsize = (width, height)
    image = cv2.resize(image, dsize)
    image = cv2.copyMakeBorder(src=image, top=5, bottom=5, left=5, right=5,
                               value=[0, 255, 0],
                               borderType=cv2.BORDER_CONSTANT)
    return image


def delete_file(path):
    """
    Delete generated file during inference
    """
    if os.path.exists(path):
        os.remove(path)


def preprocess_detection(detection):
    """
    Process yolov8 output
    """
    bboxes = detection[0].boxes
    xywhs = bboxes.xywh
    confss = bboxes.conf.unsqueeze(1)
    classes = bboxes.cls.unsqueeze(1)
    return torch.cat((xywhs, confss, classes), 1).cpu()


def argmax(listing):
    """
    Find the index of the maximum value in a list
    """
    return np.argmax(listing)


def argmin(listing):
    """
    Find the index of the minimum value in a list
    """
    return np.argmin(listing)


def get_time_now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def downscale_image(img, scale=1/3):
    """
    Downscale an OpenCV image (NumPy array) to about 1/3 of its original size.

    Args:
        img (numpy.ndarray): The input image.
        scale (float, optional): Scaling factor. Default is 1/3.

    Returns:
        numpy.ndarray: The downscaled image.
    """
    if img is None:
        raise ValueError("Input image is None")

    # Get dimensions
    height, width = img.shape[:2]

    # Compute new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize with area interpolation for better downscaling quality
    downscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return downscaled


def crop_expanded_plate(plate_xyxy, cropped_vehicle, expand_ratio=0.1):
    """
    Crops an expanded area around the given coordinates in the image.

    Args:
    plate_xyxy (tuple): A tuple containing the coordinates (x_min, y_min, x_max, y_max) of the plate.
    cropped_vehicle (numpy.ndarray): The image from which the plate is to be cropped.
    expand_ratio (float): The ratio by which to expand the cropping area on each side. Default is 0.1 (10%).

    Returns:
    numpy.ndarray: The cropped image of the expanded plate.
    """
    # Original coordinates
    x_min, y_min, x_max, y_max = plate_xyxy

    # Calculate the width and height of the original cropping area
    width = x_max - x_min
    height = y_max - y_min

    # Calculate the expansion amount (10% of the width and height by default)
    expand_x = int(expand_ratio * width)
    expand_y = int(expand_ratio * height)

    # Calculate the new coordinates with expansion
    new_x_min = max(x_min - expand_x, 0)
    new_y_min = max(y_min - expand_y, 0)
    new_x_max = min(x_max + expand_x, cropped_vehicle.shape[1])
    new_y_max = min(y_max + expand_y, cropped_vehicle.shape[0])

    # Crop the expanded area
    cropped_plate = cropped_vehicle[new_y_min:new_y_max, new_x_min:new_x_max, :]

    return cropped_plate
    
def check_legit_plate(s):
    # Remove unwanted characters
    s_cleaned = re.sub(r'[.\-\s]', '', s)

    # Regular expressions for different cases
    pattern1 = r'^[A-Za-z]{2}[0-9]{4}$'  # Matches exactly 2 letters followed by exactly 4 digits
    pattern2 = r'[A-Za-z][0-9]{4,}'      # Matches an alphabet character followed by at least 4 digits

    # Check if the cleaned string matches either pattern
    if re.search(pattern1, s_cleaned) or (re.search(pattern2, s_cleaned) and not re.match(r'^[A-Za-z]{2}', s_cleaned)):
        return True
    else:
        return False


def enhance_plate_for_paddle(plate_bgr: np.ndarray,
                             upscale_factor: float = 2.0,
                             use_adaptive_thresh: bool = True):
    """
    Enhance cropped license plate for PaddleOCR.
    """
    # Convert to grayscale for stable enhancement
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)

    # --- 1. Light contrast boost (CLAHE) ---
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    # --- 2. Gentle sharpening (unsharp mask) ---
    blur = cv2.GaussianBlur(contrast, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(contrast, 1.4, blur, -0.4, 0)

    # Convert back to 3-channel for PaddleOCR
    output = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
    return output


def rotate_clockwise(image: np.ndarray, angle_deg: float = 20):
    """
    Rotate image clockwise by given angle.
    Keeps full image (no cropping).
    """

    # Clockwise angle → negative value for cv2.getRotationMatrix2D
    angle = -abs(angle_deg)

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding dimensions to avoid cropping
    cos_val = abs(M[0, 0])
    sin_val = abs(M[0, 1])

    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))

    # Adjust matrix for translation
    M[0, 2] += (new_w // 2) - center[0]
    M[1, 2] += (new_h // 2) - center[1]

    # Apply rotation
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
    return rotated