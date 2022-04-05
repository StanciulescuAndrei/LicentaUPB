import torch.nn as nn
import torch
import torchvision.models

from ModelBuilder.ClassifierHead import DeepLabHead
from ModelBuilder.DenseNet.DenseNet import DenseNet
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter


class DenseNet201DeepLab(nn.Module):
    def __init__(self):
        super(DenseNet201DeepLab, self).__init__()

        layers = {'features': 'out'}
        model_fe = DenseNet(32, (6, 12, 48, 32), 64)
        model_fe.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.FeatureExtractor = IntermediateLayerGetter(model_fe, layers)
        self.Classifier = DeepLabHead(1920, 1, [4, 8, 12])

    def forward(self, x):
        input_shape = x.shape[-2:]
        feature_maps = self.FeatureExtractor(x)['out']
        print(feature_maps.size())
        output = self.Classifier.forward(feature_maps)
        output = F.interpolate(output, size=input_shape, mode="bilinear", align_corners=False)
        return output