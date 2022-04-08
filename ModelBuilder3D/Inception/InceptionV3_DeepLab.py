import torch.nn as nn
import torch
import torchvision.models

from ModelBuilder.ClassifierHead import DeepLabHead
from ModelBuilder.Inception.InceptionV3 import Inception3
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter


class InceptionV3DeepLab(nn.Module):
    def __init__(self):
        super(InceptionV3DeepLab, self).__init__()

        layers = {'Mixed_7c': 'out'}
        model_fe = Inception3(transform_input=False, aux_logits=False)
        self.FeatureExtractor = IntermediateLayerGetter(model_fe, layers)
        self.Classifier = DeepLabHead(2048, 1, [4, 8, 12])

    def forward(self, x):
        input_shape = x.shape[-2:]
        feature_maps = self.FeatureExtractor(x)['out']
        output = self.Classifier.forward(feature_maps)
        output = F.interpolate(output, size=input_shape, mode="bilinear", align_corners=False)
        return output
