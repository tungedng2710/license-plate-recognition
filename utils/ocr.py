import random
import string
import cv2
from torch import nn
from PIL import Image

import easyocr
from .utils import delete_file
# from vietocr.tool.predictor import Predictor
# from vietocr.tool.config import Cfg

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
        number = int(10000*number)
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


class EasyOCR(nn.Module):
    """ 
    OCR Module using EasyOCR library (https://github.com/JaidedAI/EasyOCR)
    """
    def __init__(self):
        super().__init__()
        self.reader = easyocr.Reader(["vi"])

    def forward(self, image):
        """
        Overwrite the forward method of nn.Module
        """
        cv2.imwrite("temp.jpg", image)
        result = self.reader.readtext("temp.jpg")
        delete_file("temp.jpg")
        if len(result) > 0:
            return result[0][1]
        else:
            return ''