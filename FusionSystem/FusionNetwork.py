import torch.nn as nn
import torch
import torchvision.models

from ModelBuilder.ResNet152_DeepLab import ResNet152DeepLab
from ModelBuilder.ResNeXt101_DeepLab import ResNeXt101DeepLab
from ModelBuilder.Inception.InceptionV3_DeepLab import InceptionV3DeepLab
from ModelBuilder.DenseNet.DenseNet201_DeepLab import DenseNet201DeepLab

from FusionSystem.FusionSystem import FusionSystem


class FusionNetwork(nn.Module):
    def __init__(self):
        super(FusionNetwork, self).__init__()
        self.model_list = ['DenseNet201', 'Inception', 'ResNet152', 'ResNeXt101']

        # ResNet152
        self.resnet = ResNet152DeepLab()
        checkpoint = torch.load('G:/MachineLearning/models/ResNet152/ResNet152_checkpoint.pt', map_location=torch.device('cpu'))
        self.resnet.load_state_dict(checkpoint['model_state_dict'])

        self.resnext = ResNeXt101DeepLab()
        checkpoint = torch.load('G:/MachineLearning/models/ResNeXt101/ResNeXt101_checkpoint.pt', map_location=torch.device('cpu'))
        self.resnext.load_state_dict(checkpoint['model_state_dict'])

        self.densenet = DenseNet201DeepLab()
        checkpoint = torch.load('G:/MachineLearning/models/DenseNet201/DenseNet201_checkpoint.pt', map_location=torch.device('cpu'))
        self.densenet.load_state_dict(checkpoint['model_state_dict'])

        self.inception = InceptionV3DeepLab()
        checkpoint = torch.load('G:/MachineLearning/models/Inception/Inception_checkpoint.pt', map_location=torch.device('cpu'))
        self.inception.load_state_dict(checkpoint['model_state_dict'])

        self.fusion = FusionSystem(num_sources=4)
        checkpoint = torch.load('G:/MachineLearning/models/Fusion/Fusion_checkpoint.pt', map_location=torch.device('cpu'))
        self.fusion.load_state_dict(checkpoint['model_state_dict'])


    def forward(self, x):
        resnet_output = self.resnet(x).squeeze()
        resnext_output = self.resnext(x).squeeze()
        densenet_output = self.densenet(x).squeeze()
        inception_output = self.inception(x).squeeze()
        fusion_input = torch.stack([densenet_output, inception_output, resnet_output, resnext_output], dim=1)
        return self.fusion(fusion_input)
