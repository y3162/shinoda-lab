from pathlib import Path

import torch
import torch.nn as nn


class SEModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        checkpoint_dir: str | Path,
        checkpoint_name: str,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        checkpoint_dir = Path(checkpoint_dir)
        config_path = checkpoint_dir / 'config.json'
        weight_path = checkpoint_dir / checkpoint_name
        if not config_path.is_file():
            raise FileNotFoundError(f'config not found: {config_path}')
        if not weight_path.is_file():
            raise FileNotFoundError(f'checkpoint not found: {weight_path}')

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model_name = model_name

        match model_name:
            case 'mp_senet':
                from src.mp_senet.model.model import MPNet
                from src.mp_senet.train import mag_pha_istft, mag_pha_stft
                from src.mp_senet.utils import json_to_namespace, load_checkpoint
                self.config = json_to_namespace(str(config_path))
                self.generator = MPNet(self.config.model, self.config.data.stft.n_fft)
                self._stft = mag_pha_stft
                self._istft = mag_pha_istft
            case 'semamba':
                from src.se_mamba.model.semamba import SEMamba
                from src.se_mamba.model.stfts import mag_phase_istft, mag_phase_stft
                from src.se_mamba.utils import json_to_namespace, load_checkpoint
                self.config = json_to_namespace(str(config_path))
                self.generator = SEMamba(self.config.model, self.config.data.stft.n_fft)
                self._stft = mag_phase_stft
                self._istft = mag_phase_istft
            case 'semamba_pp':
                from src.se_mamba_pp.model.semambapp import SEMambapp
                from src.se_mamba_pp.model.stfts import mag_phase_istft, mag_phase_stft
                from src.se_mamba_pp.utils import json_to_namespace, load_checkpoint
                self.config = json_to_namespace(str(config_path))
                self.generator = SEMambapp(self.config.model, self.config.data.stft.n_fft)
                self._stft = mag_phase_stft
                self._istft = mag_phase_istft
            case _:
                raise ValueError(f'Model {model_name} not found')

        state = load_checkpoint(weight_path, device)
        self.generator.load_state_dict(state['generator'])
        self.generator.to(device)
        self.generator.eval()

    @torch.inference_mode()
    def enhance(self, noisy_audio: torch.Tensor) -> torch.Tensor:
        """Enhance noisy waveform(s).

        Args:
            noisy_audio: `[B, T]` or `[T]`.

        Returns:
            Enhanced waveform with the same shape as the input, on the original
            amplitude scale.
        """
        squeeze = False
        if noisy_audio.ndim == 1:
            noisy_audio = noisy_audio.unsqueeze(0)
            squeeze = True
        elif noisy_audio.ndim != 2:
            raise ValueError(
                f'noisy_audio must be [B, T] or [T], got shape {tuple(noisy_audio.shape)}',
            )

        noisy_audio = noisy_audio.to(self.device, dtype=torch.float32)
        length = noisy_audio.size(1)
        energy = noisy_audio.pow(2).sum(dim=1).clamp_min(1e-12)
        alpha = torch.sqrt(length / energy).unsqueeze(1)
        normalized = noisy_audio * alpha

        stft = self.config.data.stft
        noisy_mag, noisy_pha, _ = self._stft(
            normalized,
            stft.n_fft,
            stft.hop_size,
            stft.win_size,
            stft.compress_factor,
        )
        mag_g, pha_g, _ = self.generator(noisy_mag, noisy_pha)
        audio_g = self._istft(
            mag_g,
            pha_g,
            stft.n_fft,
            stft.hop_size,
            stft.win_size,
            stft.compress_factor,
        )
        if audio_g.size(1) > length:
            audio_g = audio_g[:, :length]
        elif audio_g.size(1) < length:
            audio_g = torch.nn.functional.pad(audio_g, (0, length - audio_g.size(1)))

        enhanced = audio_g / alpha
        if squeeze:
            enhanced = enhanced.squeeze(0)
        return enhanced
