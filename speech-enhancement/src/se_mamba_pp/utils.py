import os

import numpy as np
import torch
from pesq import pesq


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


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
