import torch
from torch import nn


class Norm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(channels)*20.)

    def forward(self, x):
        norm = torch.norm(x, dim=1, keepdim=True)

        x = torch.div(x, norm)
        weight = self.weight.view(1, -1, 1, 1).expand_as(x)

        return weight * x
