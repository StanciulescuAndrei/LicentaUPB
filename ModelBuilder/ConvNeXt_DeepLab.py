import torch.nn as nn
import torch
import torchvision.models

from ModelBuilder.ClassifierHead import DeepLabHead
from ModelBuilder.convnext import convnext_xlarge
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter


class ConvNeXtDeepLab(nn.Module):
    def __init__(self):
        super(ConvNeXtDeepLab, self).__init__()

        self.FeatureExtractor = convnext_xlarge(in_chans=1)
        self.Classifier = DeepLabHead(2048, 1)

    def forward(self, x):
        input_shape = x.shape[-2:]
        feature_maps = self.FeatureExtractor.forward_features(x)
        output = self.Classifier.forward(feature_maps)
        output = F.interpolate(output, size=input_shape, mode="bilinear", align_corners=False)
        return output
