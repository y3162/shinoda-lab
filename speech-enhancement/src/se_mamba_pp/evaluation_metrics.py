from auraloss.freq import MultiResolutionSTFTLoss
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
import torch


def load_modules(cfg, device):
    mrstft = MultiResolutionSTFTLoss(
        sample_rate=cfg.sampling_rate,
    ).to(device)
    pesq = PerceptualEvaluationSpeechQuality(
        fs=cfg.sampling_rate,
        mode='wb',
    ).to(device)
    utmos = (
        torch.hub.load(
            'tarepan/SpeechMOS:v1.2.0',
            'utmos22_strong',
            trust_repo=True,
        )
        .to(device)
        .eval()
    )
    return mrstft, pesq, utmos


def compute_val_metrics(mrstft, pesq, utmos, clean, pred, cfg):
    mrstft_loss = mrstft(pred.unsqueeze(1), clean.unsqueeze(1))
    pesq_score = pesq(pred, clean)
    with torch.no_grad():
        utmos_score = utmos(pred, cfg.sampling_rate)
    return {
        'mrstft_score': mrstft_loss,
        'pesq_score': pesq_score,
        'utmos_score': utmos_score,
    }
