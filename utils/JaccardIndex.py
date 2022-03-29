import torch.nn as nn


class JaccardIndex(nn.Module):
    def __init__(self):
        super(JaccardIndex, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = inputs.sum() + targets.sum()
        union = total - intersection
        iou = (intersection + smooth)/(union + smooth)

        return iou
