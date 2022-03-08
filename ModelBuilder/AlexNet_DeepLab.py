import torch.nn as nn
import torch
import torchvision.models

from ModelBuilder.ClassifierHead import DeepLabHead
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter


class AlexNetDeepLab(nn.Module):
    def __init__(self):
        super(AlexNetDeepLab, self).__init__()

        layers = {'features': 'out'}
        model_fe = torchvision.models.alexnet(pretrained=True)
        self.FeatureExtractor = IntermediateLayerGetter(model_fe, layers)
        self.FeatureExtractor.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        self.Classifier = DeepLabHead(256, 1)

    def forward(self, x):
        input_shape = x.shape[-2:]
        feature_maps = self.FeatureExtractor(x)['out']
        output = self.Classifier.forward(feature_maps)
        output = F.interpolate(output, size=input_shape, mode="bilinear", align_corners=False)
        return output
