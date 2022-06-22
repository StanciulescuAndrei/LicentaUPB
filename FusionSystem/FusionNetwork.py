import torch.nn as nn
import torch
import torchvision.models

from ModelBuilder.ResNet152_DeepLab import ResNet152DeepLab
from ModelBuilder.ResNeXt101_DeepLab import ResNeXt101DeepLab
from ModelBuilder.Inception.InceptionV3_DeepLab import InceptionV3DeepLab
from ModelBuilder.DenseNet.DenseNet201_DeepLab import DenseNet201DeepLab

from FusionSystem.FusionSystem import FusionSystem


class FusionNetwork(nn.Module):
    def __init__(self, path, threshold=0.5):
        super(FusionNetwork, self).__init__()
        self.model_list = ['DenseNet201', 'Inception', 'ResNet152', 'ResNeXt101']
        self.threshold = threshold

        # ResNet152
        self.resnet = ResNet152DeepLab()
        checkpoint = torch.load(path + 'ResNet152/ResNet152_checkpoint.pt', map_location=torch.device('cpu'))
        self.resnet.load_state_dict(checkpoint['model_state_dict'])

        self.resnext = ResNeXt101DeepLab()
        checkpoint = torch.load(path + 'ResNeXt101/ResNeXt101_checkpoint.pt', map_location=torch.device('cpu'))
        self.resnext.load_state_dict(checkpoint['model_state_dict'])

        self.densenet = DenseNet201DeepLab()
        checkpoint = torch.load(path + 'DenseNet201/DenseNet201_checkpoint.pt', map_location=torch.device('cpu'))
        self.densenet.load_state_dict(checkpoint['model_state_dict'])

        self.inception = InceptionV3DeepLab()
        checkpoint = torch.load(path + 'Inception/Inception_checkpoint.pt', map_location=torch.device('cpu'))
        self.inception.load_state_dict(checkpoint['model_state_dict'])

        self.fusion = FusionSystem(num_sources=4)
        checkpoint = torch.load(path + 'Fusion/Fusion_checkpoint.pt', map_location=torch.device('cpu'))
        self.fusion.load_state_dict(checkpoint['model_state_dict'])


    def forward(self, x):
        sigmoid = nn.Sigmoid()

        resnet_output = sigmoid(self.resnet(x))
        resnet_output[resnet_output >= self.threshold] = 1
        resnet_output[resnet_output < self.threshold] = 0

        resnext_output = sigmoid(self.resnext(x))
        resnext_output[resnext_output >= self.threshold] = 1
        resnext_output[resnext_output < self.threshold] = 0

        densenet_output = sigmoid(self.densenet(x))
        densenet_output[densenet_output >= self.threshold] = 1
        densenet_output[densenet_output < self.threshold] = 0

        inception_output = sigmoid(self.inception(x))
        inception_output[inception_output >= self.threshold] = 1
        inception_output[inception_output < self.threshold] = 0

        fusion_input = torch.stack([densenet_output, inception_output, resnet_output, resnext_output], dim=1).squeeze(2)
        out = sigmoid(self.fusion(fusion_input))

        return out
