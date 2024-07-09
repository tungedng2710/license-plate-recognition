import cv2
import numpy as np


class Resize:
    def __init__(self, limit_side_len):
        self.limit_side_len = limit_side_len

    def __call__(self, img):
        src_h, src_w = img.shape[:2]
        ratio = 1.
        if max(src_h, src_w) > self.limit_side_len:
            ratio = self.limit_side_len / src_h if src_h > src_w else self.limit_side_len / src_w
            img = cv2.resize(img, (round(ratio * src_w), round(ratio * src_h)))

        return {
            'image': cv2.copyMakeBorder(
                img,
                0, self.limit_side_len - img.shape[0], 0, self.limit_side_len - img.shape[1],
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            ),
            'shape': np.array([src_h, src_w, ratio]),
        }

class Normalize:
    def __init__(self, mean, std, scale=1/255):
        self.scale = np.float32(scale)
        shape = (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        data['image'] = (data['image'].astype('float32') * self.scale - self.mean) / self.std
        return data


class HWCToCHW:
    def __call__(self, data):
        data['image'] = data['image'].transpose((2, 0, 1))
        return data


class PickKeys:
    def __init__(self, *keys):
        self.keys = keys

    def __call__(self, data):
        return [data[key] for key in self.keys]
