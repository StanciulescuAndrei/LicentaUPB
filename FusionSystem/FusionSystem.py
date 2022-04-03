import torch.nn as nn
import torch


class FusionSystem(nn.Module):
    def __init__(self, num_sources=4):
        super().__init__()
        self.num_sources = num_sources
        self.weights = nn.Parameter(torch.ones(num_sources) * 1.0 / num_sources)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        minibatch_size = x.size()[0]
        image_size_x = x.size()[2]
        image_size_y = x.size()[3]
        out = torch.zeros(minibatch_size, image_size_x, image_size_y, device=torch.device('cuda'))
        assert x.size()[1] == self.num_sources, "Incorrect number of segmentation channels"
        for i in range(self.num_sources):
            out[:, :, :] += self.weights[i] * x[:, i, :, :]
        return torch.round(self.activation(out))
