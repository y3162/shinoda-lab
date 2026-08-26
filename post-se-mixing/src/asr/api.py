import torch
import torch.nn as nn


class ASRModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model_name = model_name

        match model_name:
            case 'parakeet-tdt-0.6b-v2':
                from src.asr.parakeet_tdt_0_6b_v2 import ParakeetTDT06BV2
                self._model = ParakeetTDT06BV2()
            case 'whisper-large-v3':
                from src.asr.whisper_large_v3 import WhisperLargeV3
                self._model = WhisperLargeV3()
            case _:
                raise ValueError(f'Model {model_name} not found')

        self._model.to(device)
        self._model.eval()
        self._model.greedy_mode()

    @torch.inference_mode()
    def transcribe(
        self,
        audio: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> str | list[str]:
        """Transcribe waveform(s).

        Args:
            audio: `[B, T]` or `[T]`, 16 kHz float waveform.
            lengths: Optional valid sample counts for a padded `[B, T]` batch.

        Returns:
            Transcript string for a single waveform, or a list of transcripts for
            a batch.
        """
        squeeze = False
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
            squeeze = True
        elif audio.ndim != 2:
            raise ValueError(
                f'audio must be [B, T] or [T], got shape {tuple(audio.shape)}',
            )

        audio = audio.to(self.device, dtype=torch.float32)
        if lengths is None:
            lengths = torch.full(
                (audio.size(0),),
                audio.size(1),
                dtype=torch.long,
                device=self.device,
            )
        else:
            lengths = lengths.to(self.device, dtype=torch.long)

        outputs = self._model(audio, lengths)
        texts = [hypotheses[0].text for hypotheses in outputs.results]
        if squeeze:
            return texts[0]
        return texts
