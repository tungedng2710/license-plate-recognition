import random
import string
import torch
from torch import nn

class DummyModel(nn.Module):
    """
    I'm just a Dummy model for filling the gap
    Replace me with an OCR model
    """
    def __init__(self):
        super().__init__()
        print("You are using dummy OCR model!")

    def forward(self, input):
        number = random.uniform(1, 9)
        number = int(10000*number)
        number = str(number)
        letter = random.choice(string.ascii_uppercase)
        dummy_output = "30"+letter+number
        return dummy_output