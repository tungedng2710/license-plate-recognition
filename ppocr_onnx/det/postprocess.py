import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper


class DBPostProcess:
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self, thresh=0.3):
        self.thresh = thresh
        self.min_size = 3

    def boxes_from_bitmap(self, pred, bitmap, shape, unclip_ratio, box_thresh):
        '''
        bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        contours = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]

        boxes = []
        scores = []
        for contour in contours:
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score(pred, points.reshape(-1, 2))
            if box_thresh > score:
                continue

            box = self.unclip(points, unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(np.round(box[:, 0] / shape[2]), 0, shape[1])
            box[:, 1] = np.clip(np.round(box[:, 1] / shape[2]), 0, shape[0])
            boxes.append(box)
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def unclip(self, box, unclip_ratio):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(cv2.boxPoints(bounding_box), key=lambda x: x[0])

        index = [0, 2, 3, 1]
        if points[1][1] < points[0][1]:
            index[0], index[3] = index[3], index[0]
        if points[3][1] < points[2][1]:
            index[1], index[2] = index[2], index[1]

        return [points[i] for i in index], min(bounding_box[1])

    def box_score(self, bitmap, _box):
        '''
        box_score: use bbox mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs, shape_list, unclip_ratio, box_thresh):
        return [
            self.boxes_from_bitmap(
                pred,
                cv2.dilate((pred > self.thresh).astype(np.uint8), np.array([[1, 1], [1, 1]])),
                shape, unclip_ratio, box_thresh
            )[0]
            for shape, pred in zip(
                shape_list,
                outs[:, 0, :, :]
            )
        ]
