import torch.nn as nn
import torch
import torchvision.models

from ModelBuilder.ClassifierHead import DeepLabHead
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter


class InceptionV3DeepLab(nn.Module):
    def __init__(self):
        super(InceptionV3DeepLab, self).__init__()

        layers = {'Mixed_7c': 'out'}
        model_fe = torchvision.models.inception_v3(pretrained=True, transform_input=False, aux_logits=False)
        self.FeatureExtractor = IntermediateLayerGetter(model_fe, layers)
        self.FeatureExtractor.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.Classifier = DeepLabHead(2048, 1)

    def forward(self, x):
        input_shape = x.shape[-2:]
        feature_maps = self.FeatureExtractor(x)['out']
        output = self.Classifier.forward(feature_maps)
        output = F.interpolate(output, size=input_shape, mode="bilinear", align_corners=False)
        return output
