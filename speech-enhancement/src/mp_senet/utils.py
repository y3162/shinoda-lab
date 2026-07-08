import os

import numpy as np
import torch
import torch.nn as nn
from pesq import pesq


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


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


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    return torch.load(filepath, map_location=device)


def save_checkpoint(filepath, obj):
    torch.save(obj, filepath)


def cal_pesq(clean, noisy, sr=16000):
    try:
        score = pesq(sr, clean, noisy, 'wb')
    except Exception:
        score = -1
    return score


def batch_pesq(clean, noisy):
    scores = np.array([
        cal_pesq(clean_utt, noisy_utt)
        for clean_utt, noisy_utt in zip(clean, noisy)
    ])
    if -1 in scores:
        return None
    scores = (scores - 1) / 3.5
    return torch.FloatTensor(scores)


def pesq_score(utts_r, utts_g, h):
    scores = [
        cal_pesq(
            utts_r[i].squeeze().cpu().numpy(),
            utts_g[i].squeeze().cpu().numpy(),
            h.sampling_rate,
        )
        for i in range(len(utts_r))
    ]
    return np.mean(scores)
