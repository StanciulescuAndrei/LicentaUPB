import torch.nn as nn
import torch


class FusionSystem(nn.Module):
    def __init__(self, num_sources=4):
        super().__init__()
        self.num_sources = num_sources
        self.weights = nn.Conv2d(in_channels=num_sources, out_channels=1, kernel_size=(1, 1), stride=1, padding=0)
        self.activation = nn.Sigmoid()

    def forward(self, x):

        assert x.size()[1] == self.num_sources, "Incorrect number of segmentation channels"
        out = self.weights(x.float())
        return out
