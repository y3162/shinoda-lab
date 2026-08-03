from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from pesq import pesq


def pad_collate(batch):
    cleans, noisys = zip(*batch)
    lengths = torch.tensor([int(c.shape[-1]) for c in cleans], dtype=torch.long)
    max_len = int(lengths.max().item())
    batch_size = len(batch)
    clean_batch = cleans[0].new_zeros((batch_size, max_len))
    noisy_batch = noisys[0].new_zeros((batch_size, max_len))
    for i, (clean, noisy) in enumerate(zip(cleans, noisys)):
        clean_len = int(clean.shape[-1])
        noisy_len = int(noisy.shape[-1])
        clean_batch[i, :clean_len] = clean
        noisy_batch[i, :noisy_len] = noisy
    return clean_batch, noisy_batch, lengths


def _pesq_one(args):
    clean, enhanced, sr = args
    try:
        return float(pesq(sr, clean, enhanced, 'wb'))
    except Exception:
        return -1.0


def compute_pesq_parallel(refs, ests, sr, num_workers):
    if not refs:
        return 0.0, 0
    payloads = [
        (np.asarray(ref, dtype=np.float64), np.asarray(est, dtype=np.float64), int(sr))
        for ref, est in zip(refs, ests)
    ]
    workers = max(1, int(num_workers))
    if workers == 1 or len(payloads) == 1:
        scores = [_pesq_one(payload) for payload in payloads]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            scores = list(executor.map(_pesq_one, payloads))
    return float(sum(scores)), len(scores)


def deterministic_noise_seed(seed, index):
    return (int(seed) * 1_000_003 + int(index)) & 0xFFFFFFFFFFFFFFFF


def collect_unpadded_waveforms(clean_audio, audio_g, lengths):
    refs = []
    ests = []
    for i in range(clean_audio.size(0)):
        length = int(lengths[i].item())
        ref = clean_audio[i, :length]
        est = audio_g[i]
        use_len = min(length, int(est.shape[-1]))
        refs.append(ref[:use_len].detach().cpu().numpy())
        ests.append(est[:use_len].detach().cpu().numpy())
    return refs, ests
