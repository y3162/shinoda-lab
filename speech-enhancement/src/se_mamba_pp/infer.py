import argparse
import logging
import os
import json
import shutil
import time
from pathlib import Path

import librosa
import soundfile as sf
import torch

from src.se_mamba_pp.model.semambapp import SEMambapp
from src.se_mamba_pp.model.stfts import mag_phase_istft, mag_phase_stft
from src.se_mamba_pp.utils import AttrDict, load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)


def enhance_waveform(model, h, noisy_wav, device):
    noisy_wav = torch.FloatTensor(noisy_wav).to(device)
    norm_factor = torch.sqrt(
        torch.tensor(len(noisy_wav), dtype=torch.float32, device=device)
        / torch.sum(noisy_wav ** 2.0).clamp(min=1e-8)
    )
    noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)

    with torch.no_grad():
        noisy_mag, noisy_pha, _ = mag_phase_stft(
            noisy_wav,
            h.n_fft,
            h.hop_size,
            h.win_size,
            h.compress_factor,
        )
        mag_g, pha_g, _ = model(noisy_mag, noisy_pha)
        audio_g = mag_phase_istft(
            mag_g,
            pha_g,
            h.n_fft,
            h.hop_size,
            h.win_size,
            h.compress_factor,
        )

    return (audio_g / norm_factor).squeeze().cpu().numpy()


def load_model(config_path, checkpoint_path, device):
    with open(config_path) as f:
        h = AttrDict(json.loads(f.read()))
    model = SEMambapp(h).to(device)
    ckpt = load_checkpoint(checkpoint_path, device)
    if 'generator' in ckpt:
        model.load_state_dict(ckpt['generator'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, h


def infer_one(model, h, input_wav, output_wav, device, log):
    log.info('Loading audio: %s', input_wav)
    noisy_wav, _ = librosa.load(input_wav, sr=h.sampling_rate)
    duration = len(noisy_wav) / h.sampling_rate

    t0 = time.time()
    restored = enhance_waveform(model, h, noisy_wav, device)
    elapsed = time.time() - t0
    log.info(
        'Inference finished in %.3fs (RTF: %.3f)',
        elapsed,
        elapsed / duration,
    )

    os.makedirs(os.path.dirname(output_wav) or '.', exist_ok=True)
    sf.write(output_wav, restored, h.sampling_rate, subtype='PCM_16')
    log.info('Restored audio saved to: %s', output_wav)


def main():
    parser = argparse.ArgumentParser(description='SEMamba++ inference')
    parser.add_argument('--input_wav', default=None)
    parser.add_argument('--output_wav', default=None)
    parser.add_argument('--input_dir', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument(
        '--checkpoint',
        default='data/checkpoints/se_mamba_pp/pretrained/semambapp.pth',
    )
    parser.add_argument(
        '--config',
        default='src/se_mamba_pp/configs/default.json',
    )
    parser.add_argument('--copy_input', action='store_true')
    args = parser.parse_args()

    log = logging.getLogger('SEMamba++')

    if args.input_wav is None and args.input_dir is None:
        parser.error('Provide --input_wav or --input_dir')
    if args.input_wav is not None and args.output_wav is None:
        parser.error('--output_wav is required with --input_wav')
    if args.input_dir is not None and args.output_dir is None:
        parser.error('--output_dir is required with --input_dir')

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for SEMamba++ inference.')

    device = torch.device('cuda:0')
    log.info('Using device: %s', device)
    log.info('Loading config from: %s', args.config)
    log.info('Loading checkpoint from: %s', args.checkpoint)

    model, h = load_model(args.config, args.checkpoint, device)
    num_params = sum(p.numel() for p in model.parameters())
    log.info('Model parameters: %s', f'{num_params:,}')

    if args.input_wav is not None:
        infer_one(model, h, args.input_wav, args.output_wav, device, log)
        return

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_paths = sorted(input_dir.glob('*.wav'))
    if not wav_paths:
        raise FileNotFoundError('No .wav files found in {}'.format(input_dir))

    for wav_path in wav_paths:
        out_path = output_dir / '{}_restored.wav'.format(wav_path.stem)
        infer_one(model, h, str(wav_path), str(out_path), device, log)
        if args.copy_input:
            shutil.copy2(wav_path, output_dir / wav_path.name)


if __name__ == '__main__':
    main()
