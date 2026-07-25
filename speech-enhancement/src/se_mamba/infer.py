import argparse
from pathlib import Path

import librosa
import soundfile as sf
import torch

from src.se_mamba.model.pcs400 import cal_pcs
from src.se_mamba.model.semamba import SEMamba
from src.se_mamba.model.stfts import mag_phase_istft, mag_phase_stft
from src.se_mamba.utils import json_to_namespace, load_checkpoint


def _str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', '1'):
        return True
    if value.lower() in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def resolve_checkpoint(checkpoint):
    path = Path(checkpoint)
    if path.is_file():
        return path, path.parent / 'config.json'
    for name in ('g_best', 'g_latest'):
        candidate = path / name
        if candidate.is_file():
            return candidate, path / 'config.json'
    raise FileNotFoundError(f'No g_best/g_latest under {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--input_folder', required=True)
    parser.add_argument('--output_folder', required=True)
    parser.add_argument('--post_processing_pcs', type=_str2bool, default=False)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for inference.')
    device = torch.device('cuda')

    ckpt_path, default_config = resolve_checkpoint(args.checkpoint)
    config_path = Path(args.config) if args.config is not None else default_config
    if not config_path.is_file():
        raise FileNotFoundError(f'Config not found: {config_path}')
    config = json_to_namespace(str(config_path))
    stft = config.data.stft

    model = SEMamba(config.model, stft.n_fft).to(device)
    state = load_checkpoint(ckpt_path, device)
    model.load_state_dict(state['generator'])
    model.eval()

    input_dir = Path(args.input_folder)
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for fname in sorted(input_dir.iterdir()):
            if not fname.is_file():
                continue
            noisy_wav, _ = librosa.load(str(fname), sr=config.data.sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)
            norm_factor = torch.sqrt(
                len(noisy_wav) / torch.sum(noisy_wav ** 2.0),
            ).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            noisy_amp, noisy_pha, _ = mag_phase_stft(
                noisy_wav, stft.n_fft, stft.hop_size, stft.win_size, stft.compress_factor,
            )
            amp_g, pha_g, _ = model(noisy_amp, noisy_pha)
            audio_g = mag_phase_istft(
                amp_g, pha_g, stft.n_fft, stft.hop_size, stft.win_size, stft.compress_factor,
            )
            audio_g = audio_g / norm_factor
            audio_np = audio_g.squeeze().cpu().numpy()
            if args.post_processing_pcs:
                audio_np = cal_pcs(audio_np)
            sf.write(str(output_dir / fname.name), audio_np, config.data.sampling_rate, 'PCM_16')
            print(fname.name)


if __name__ == '__main__':
    main()
