import torch


def mix_linear(
    noisy: torch.Tensor,
    enhanced: torch.Tensor,
    coeff: float,
) -> torch.Tensor:
    length = min(noisy.shape[-1], enhanced.shape[-1])
    return coeff * noisy[..., :length] + (1.0 - coeff) * enhanced[..., :length]
