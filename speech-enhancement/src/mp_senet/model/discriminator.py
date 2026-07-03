import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from pesq import pesq
from torch.nn.utils import spectral_norm
from src.mp_senet.utils import LearnableSigmoid1d


def cal_pesq(clean, noisy, sr=16000):
    try:
        score = pesq(sr, clean, noisy, "wb")
    except:
        # error can happen due to silent period
        score = -1

    return score


def batch_pesq(clean, noisy):
    scores = Parallel(n_jobs=15)(
        delayed(cal_pesq)(clean_utt, noisy_utt)
        for clean_utt, noisy_utt in zip(clean, noisy)
    )

    scores = np.array(scores)

    if -1 in scores:
        return None

    scores = (scores - 1) / 3.5

    return torch.FloatTensor(scores)


def metric_loss(metric_ref, metrics_gen):
    loss = 0

    for metric_gen in metrics_gen:
        loss = loss + F.mse_loss(
            metric_ref,
            metric_gen.flatten(),
        )

    return loss


class MetricDiscriminator(nn.Module):
    def __init__(self, dim=16, in_channel=2):
        super().__init__()

        self.layers = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channel,
                    dim,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                )
            ),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),

            spectral_norm(
                nn.Conv2d(
                    dim,
                    dim * 2,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                )
            ),
            nn.InstanceNorm2d(dim * 2, affine=True),
            nn.PReLU(dim * 2),

            spectral_norm(
                nn.Conv2d(
                    dim * 2,
                    dim * 4,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                )
            ),
            nn.InstanceNorm2d(dim * 4, affine=True),
            nn.PReLU(dim * 4),

            spectral_norm(
                nn.Conv2d(
                    dim * 4,
                    dim * 8,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                )
            ),
            nn.InstanceNorm2d(dim * 8, affine=True),
            nn.PReLU(dim * 8),

            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),

            spectral_norm(
                nn.Linear(dim * 8, dim * 4)
            ),
            nn.Dropout(0.3),
            nn.PReLU(dim * 4),

            spectral_norm(
                nn.Linear(dim * 4, 1)
            ),
            LearnableSigmoid1d(1),
        )

    def forward(self, x, y):
        xy = torch.stack((x, y), dim=1)
        return self.layers(xy)
