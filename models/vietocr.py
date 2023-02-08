import cv2
from torch import nn
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


class VietOCR(nn.Module):
    """
    OCR Module using VietOCR library
    """
    def __init__(self):
        super().__init__()
        config = Cfg.load_config_from_name("vgg_transformer")
        config["cnn"]["pretrained"]=False
        config["device"] = "cuda:0"
        self.detector = Predictor(config)

    def forward(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        return self.detector.predict(img_pil)
