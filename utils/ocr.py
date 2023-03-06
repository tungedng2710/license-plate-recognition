import random
import string
import cv2
from torch import nn
from PIL import Image

# import easyocr
# from vietocr.tool.predictor import Predictor
# from vietocr.tool.config import Cfg
from .utils import delete_file

import os
from PIL import Image
import cv2
import numpy as np
import math
import time
import traceback
from paddleocr.ppocr.postprocess import build_post_process
import paddleocr.tools.infer.utility as utility
from paddleocr.ppocr.utils.logging import get_logger

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

logger = get_logger()


class TextRecognizer(object):
    def __init__(self, args):
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorithm = args.rec_algorithm
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_dict_path": 'en_dict.txt',
            "use_space_char": args.use_space_char
        }

        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'rec', logger)
        self.benchmark = args.benchmark
        self.use_onnx = args.use_onnx

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape

        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        if self.rec_algorithm == 'RARE':
            if resized_w > self.rec_image_shape[2]:
                resized_w = self.rec_image_shape[2]
            imgW = self.rec_image_shape[2]
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        st = time.time()

        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            # max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            self.input_tensor.copy_from_cpu(norm_img_batch)
            self.predictor.run()
            outputs = []
            for output_tensor in self.output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)
            if self.benchmark:
                self.autolog.times.stamp()
            if len(outputs) != 1:
                preds = outputs
            else:
                preds = outputs[0]

            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]

        return rec_res, time.time() - st


class PPOCR(nn.Module):
    """
    OCR Module using PPOCR library (https://github.com/PaddlePaddle/PaddleOCR)
    """

    def __init__(self):
        super().__init__()
        args = utility.parse_args()
        args.rec_model_dir = "weights/rec_ppocr_0.948/"
        self.text_recognizer = TextRecognizer(args)

    def forward(self, img_list):
        """
        Overwrite the forward method of nn.Module
        """
        img_list = [img_list]
        try:
            rec_res, _ = self.text_recognizer(img_list)
            return rec_res[0][0], rec_res[0][1]
        except Exception as E:
            logger.info(traceback.format_exc())
            logger.info(E)
            exit()
        return [], []


class DummyOCR(nn.Module):
    """
    I'm just a Dummy model for filling the gap
    Replace me with an OCR model
    """

    def __init__(self):
        super().__init__()
        print("You are using dummy OCR model!")

    def forward(self, image):
        """
        Overwrite the forward method of nn.Module
        Generate a random string
        """
        dummy_output = image
        number = random.uniform(1, 9)
        number = int(10000 * number)
        number = str(number)
        letter = random.choice(string.ascii_uppercase)
        dummy_output = f"30{letter}{number}"
        return dummy_output


# class VietOCR(nn.Module):
#     """
#     OCR Module using VietOCR library (https://github.com/pbcquoc/vietocr)
#     """
#     def __init__(self):
#         super().__init__()
#         config = Cfg.load_config_from_name("vgg_transformer")
#         config["cnn"]["pretrained"]=False
#         config["device"] = "cuda:0"
#         self.detector = Predictor(config)

#     def forward(self, img):
#         """
#         Overwrite the forward method of nn.Module
#         """
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img_pil = Image.fromarray(img)
#         return self.detector.predict(img_pil)


# class EasyOCR(nn.Module):
#     """
#     OCR Module using EasyOCR library (https://github.com/JaidedAI/EasyOCR)
#     """
#
#     def __init__(self):
#         super().__init__()
#         self.reader = easyocr.Reader(["vi"])
#
#     def forward(self, image):
#         """
#         Overwrite the forward method of nn.Module
#         """
#         cv2.imwrite("temp.jpg", image)
#         result = self.reader.readtext("temp.jpg")
#         delete_file("temp.jpg")
#         if len(result) > 0:
#             return result[0][1]
#         else:
#             return ''
