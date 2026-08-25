import argparse
import json

import torch
import torch.nn as nn


class LearnableSigmoid1d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


def json_to_namespace(json_path):
    with open(json_path, 'r') as f:
        return __json_to_namespace(json.load(f))


def __json_to_namespace(value):
    if isinstance(value, dict):
        return argparse.Namespace(**{
            key: __json_to_namespace(val) for key, val in value.items()
        })
    if isinstance(value, list):
        return [__json_to_namespace(item) for item in value]
    return value


def load_checkpoint(filepath, device):
    return torch.load(filepath, map_location=device, weights_only=False)
