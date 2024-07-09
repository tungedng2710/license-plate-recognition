import copy
from functools import cmp_to_key

import cv2
import numpy as np

from .det import TextDetector
from .rec import TextRecognizer


def perspective_crop(img, points):
    w = round(max(
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[2] - points[3]),
    ))
    h = round(max(
        np.linalg.norm(points[0] - points[3]),
        np.linalg.norm(points[1] - points[2]),
    ))
    return cv2.warpPerspective(
        img,
        cv2.getPerspectiveTransform(
            points,
            np.float32([[0, 0],[w, 0],[w, h],[0, h]])
        ),
        (w, h),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )


class DetAndRecONNXPipeline:
    def __init__(self,
                 box_thresh=0.6, 
                 unclip_ratio=1.6,
                 text_det_onnx_model="",
                 text_rec_onnx_model="",
                 text_rec_dict="ppocr_keys_v1.txt"):
        self.text_detector = TextDetector(
            text_det_onnx_model,
            box_thresh=box_thresh,
            unclip_ratio=unclip_ratio,
        )
        self.text_recognizer = TextRecognizer(text_rec_onnx_model, text_rec_dict)

    def detect_and_ocr(self, img, drop_score=0.5, unclip_ratio=None, box_thresh=None):
        ori_im = img.copy()
        dt_boxes, _ = self.text_detector(img, unclip_ratio, box_thresh)
        if dt_boxes is None:
            return []
        img_crop_list = []

        dt_boxes = sorted(
            dt_boxes,
            key=cmp_to_key(lambda x, y:
                x[0][0] - y[0][0]
                if -10 < x[0][1] - y[0][1] < 10 else
                x[0][1] - y[0][1]
            )
        )

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = perspective_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        rec_res, _ = self.text_recognizer(img_crop_list)
        res = []
        for box, rec_reuslt, img_crop in zip(dt_boxes, rec_res, img_crop_list):
            text, score = rec_reuslt
            if score >= drop_score:
                res.append(BoxedResult(box, img_crop, text, score))
        return res


class BoxedResult:
    def __init__(self, box, img, text, score):
        self.box = box
        self.img = img
        self.text = text
        self.score = score

    def __repr__(self):
        return f'{type(self).__name__}[{self.text}, {self.score}]'
