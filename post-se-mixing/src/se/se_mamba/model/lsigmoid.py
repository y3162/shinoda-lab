import torch
import torch.nn as nn


class LearnableSigmoid1D(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requires_grad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class LearnableSigmoid2D(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requires_grad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
