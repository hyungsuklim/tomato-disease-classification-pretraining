"""
"""
import torch
from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet



class PestClassifier(nn.Module):
    def __init__(self, num_class):
        super(PestClassifier, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_class)
        self.softmax = nn.Softmax(dim=1)
                     
    def forward(self, input_img):
        x = self.model(input_img)
        x = self.softmax(x)
        return x