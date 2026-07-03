import os
import glob
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(4, 3))

    im = ax.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        interpolation="none",
    )
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def get_padding_2d(kernel_size, dilation=(1, 1)):
    padding_h = int((kernel_size[0] * dilation[0] - dilation[0]) / 2)
    padding_w = int((kernel_size[1] * dilation[1] - dilation[1]) / 2)

    return padding_h, padding_w


class LearnableSigmoid1d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()

        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))

        # 元コードの記述を維持
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()

        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))

        # 元コードの記述を維持
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class Sigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()

        self.beta = beta
        self.slope = torch.ones(in_features, 1)

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class PLSigmoid(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.beta = nn.Parameter(torch.ones(in_features, 1) * 2.0)
        self.slope = nn.Parameter(torch.ones(in_features, 1))

        # 元コードの記述を維持
        self.beta.requiresGrad = True
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)

    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")

    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + "????????")
    checkpoint_paths = glob.glob(pattern)

    if len(checkpoint_paths) == 0:
        return None

    return sorted(checkpoint_paths)[-1]
