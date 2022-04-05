import torch.nn as nn

from ModelBuilder.ClassifierHead import DeepLabHead
from ModelBuilder.ConvNeXt.convnext import ConvNeXt
from torch.nn import functional as F


class ConvNeXtDeepLab(nn.Module):
    def __init__(self):
        super(ConvNeXtDeepLab, self).__init__()

        self.FeatureExtractor = ConvNeXt(in_chans=1, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
        self.Classifier = DeepLabHead(1024, 1, [4, 8, 12])

    def forward(self, x):
        input_shape = x.shape[-2:]
        feature_maps = self.FeatureExtractor.forward_features(x)
        print(feature_maps.size())
        output = self.Classifier.forward(feature_maps)
        output = F.interpolate(output, size=input_shape, mode="bilinear", align_corners=False)
        return output
