import numpy as np
import time
import onnxruntime as ort
from .preprocess import Resize, Normalize, HWCToCHW, PickKeys
from .postprocess import DBPostProcess

class OperatorGroup:
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, data):
        for op in self.ops:
            data = op(data)
        return data
    
class TextDetector:
    def __init__(self, model_path, box_thresh=0.6, unclip_ratio=1.6):
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio

        self.preprocess_op = OperatorGroup(
            Resize(limit_side_len=960),
            Normalize(std=[0.229, 0.224, 0.225], mean=[0.485, 0.456, 0.406]),
            HWCToCHW(),
            PickKeys('image', 'shape')
        )
        self.postprocess_op = DBPostProcess(thresh=0.3)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(model_path, so, providers=['CUDAExecutionProvider'])
        self.output_tensors = None
        self.predictor, self.input_tensor = sess, sess.get_inputs()[0]

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype=np.float32)
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            if np.linalg.norm(box[0] - box[1]) < 4 or np.linalg.norm(box[0] - box[3]) < 4:
                continue
            dt_boxes_new.append(box)
        return np.array(dt_boxes_new)

    def __call__(self, img, unclip_ratio=None, box_thresh=None):
        if unclip_ratio is None:
            unclip_ratio = self.unclip_ratio
        if box_thresh is None:
            box_thresh = self.box_thresh
        img, shape = self.preprocess_op(img)
    
        starttime = time.time()
        input_dict = {self.input_tensor.name: img[None]}

        outputs = self.predictor.run(self.output_tensors, input_dict)

        post_result = self.postprocess_op(outputs[0], shape[None], unclip_ratio, box_thresh)
        dt_boxes = post_result[0]
        dt_boxes = self.filter_tag_det_res(dt_boxes, shape[:2])
        elapse = time.time() - starttime
        return dt_boxes, elapse
    
    # def __call__(self, images, unclip_ratio=None, box_thresh=None):
    #     if unclip_ratio is None:
    #         unclip_ratio = self.unclip_ratio
    #     if box_thresh is None:
    #         box_thresh = self.box_thresh
    #     if not isinstance(images, list):
    #         images, shape = self.preprocess_op(images)
        
    #         # starttime = time.time()
    #         input_dict = {self.input_tensor.name: images[None]}

    #         outputs = self.predictor.run(self.output_tensors, input_dict)

    #         post_result = self.postprocess_op(outputs[0], shape[None], unclip_ratio, box_thresh)
    #         dt_boxes = post_result[0]
    #         dt_boxes = self.filter_tag_det_res(dt_boxes, shape[:2])
    #         # elapse = time.time() - starttime
    #         return dt_boxes, 0
    #     else:
    #         processed_images, shapes = [], []
    #         for image in images:
    #             image, shape = self.preprocess_op(image)
    #             processed_images.append(image)
    #             shapes.append(shape)
    #         input_dict = {self.input_tensor.name: np.stack(processed_images)}

    #         outputs = self.predictor.run(self.output_tensors, input_dict)

    #         post_results = self.postprocess_op(outputs[0], shapes, unclip_ratio, box_thresh)
    #         dt_boxes_list = []
    #         for result in post_results:
    #             dt_boxes_list.append(self.filter_tag_det_res(result, shape[:2]))
    #         return dt_boxes_list, 0