import os
import cv2

BGR_COLORS = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "amber": (0, 191, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0)
}
# VEHICLES = ["bicycle", "bus", "car", "motorbike", "person", "truck"]
VEHICLES = ['bicycle', 'bus', 'car', 'motorcycle', 'truck']

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
    bg_h = int(text_h * 1.3) # expand the background height a bit
    cv2.rectangle(img, pos, (x + text_w, y + bg_h), text_color_bg, -1)
    pos = (x, y + text_h + font_scale)
    cv2.putText(img, text, pos, font, font_scale, text_color, font_thickness)

def draw_box(img, pt1, pt2, color, thickness, r, d):
    """
    Draw more fancy bounding box
    """
    x1,y1 = pt1
    x2,y2 = pt2
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
    return cv2.resize(image, dsize)

def delete_file(path):
    """
    Delete generated file during inference
    """
    if os.path.exists(path):
        os.remove(path)