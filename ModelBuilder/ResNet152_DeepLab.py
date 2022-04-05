import torch.nn as nn
import torch
import torchvision.models

from ModelBuilder.ClassifierHead import DeepLabHead
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter


class ResNet152DeepLab(nn.Module):
    def __init__(self):
        super(ResNet152DeepLab, self).__init__()

        layers = {'layer4': 'out'}
        model_fe = torchvision.models.resnet152(pretrained=True, replace_stride_with_dilation=[False, True, True])
        model_fe.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.FeatureExtractor = IntermediateLayerGetter(model_fe, layers)
        self.Classifier = DeepLabHead(2048, 1)

    def forward(self, x):
        input_shape = x.shape[-2:]
        feature_maps = self.FeatureExtractor(x)['out']
        output = self.Classifier.forward(feature_maps)
        output = F.interpolate(output, size=input_shape, mode="bilinear", align_corners=False)
        return output
