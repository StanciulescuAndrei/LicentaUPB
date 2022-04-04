import torch.nn as nn
import torch
import torchvision.models

from ModelBuilder.ClassifierHead import DeepLabHead
from ModelBuilder.convnext import ConvNeXt
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter


class ConvNeXtDeepLab(nn.Module):
    def __init__(self):
        super(ConvNeXtDeepLab, self).__init__()

        self.FeatureExtractor = ConvNeXt(in_chans=1, depths=[3, 9, 18, 27], dims=[256, 512, 1024, 2048])
        self.Classifier = DeepLabHead(2048, 1)

    def forward(self, x):
        input_shape = x.shape[-2:]
        feature_maps = self.FeatureExtractor.forward_features(x)
        # print(feature_maps.size())
        output = self.Classifier.forward(feature_maps)
        output = F.interpolate(output, size=input_shape, mode="bilinear", align_corners=False)
        return output
