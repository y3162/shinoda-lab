from typing import List
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.utils import logging as hf_logging

from src.api.inference_preprocessor import (
    WhisperInferencePreprocessor,
    WHISPER_SHORT_FORM_MEL_FRAMES,
)
from src.config import DEFAULT_SAMPLE_RATE
from src.utils.print import print_log


hf_logging.set_verbosity_error()


class WhisperLargeV3(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            'openai/whisper-large-v3',
            low_cpu_mem_usage=True,
            use_safetensors=True,
            torch_dtype=dtype,
            attn_implementation='flash_attention_2',
        )
        self.processor = AutoProcessor.from_pretrained('openai/whisper-large-v3')

        self.num_beams = 1
        self.return_best_hypothesis = True

        self.model.generation_config.language = 'en'
        self.model.generation_config.task = 'transcribe'

    @property
    def __return_best(self) -> bool:
        return self.num_beams == 1 or self.return_best_hypothesis

    def greedy_mode(
        self,
    ):
        self.num_beams = 1
        self.return_best_hypothesis = True

    def beam_mode(
        self,
        num_beams: int,
        strategy: str = 'beam',
        return_best_hypothesis: bool = False,
    ):
        if strategy != 'beam':
            raise ValueError(
                f'WhisperLargeV3 only supports strategy="beam", but got {strategy!r}'
            )

        self.num_beams = num_beams
        self.return_best_hypothesis = return_best_hypothesis

    def create_inference_preprocessor(
        self,
    ) -> WhisperInferencePreprocessor:
        return WhisperInferencePreprocessor(self.processor)

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
    ) -> SimpleNamespace:

        inputs = self.__preprocess(waveforms, lengths)
        return self.forward_from_features(inputs.input_features, inputs.attention_mask)

    def __trim_to_valid_mel_frames(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_frames = int(attention_mask.sum(dim=-1).max().item())
        return input_features[..., :valid_frames], attention_mask[..., :valid_frames]

    def forward_from_features(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        input_features = input_features.to(device=device, dtype=dtype)
        attention_mask = attention_mask.to(device=device)

        mel_lengths = attention_mask.sum(dim=-1)
        short_indices = (mel_lengths <= WHISPER_SHORT_FORM_MEL_FRAMES).nonzero(as_tuple=True)[0]
        long_indices = (mel_lengths > WHISPER_SHORT_FORM_MEL_FRAMES).nonzero(as_tuple=True)[0]

        results: List = [None] * input_features.size(0)

        if short_indices.numel() > 0:
            short_features, short_mask = self.__trim_to_valid_mel_frames(
                input_features[short_indices],
                attention_mask[short_indices],
            )
            short_outputs = self.__generate_features(
                short_features,
                short_mask,
                return_timestamps=False,
            )
            for idx, result in zip(short_indices.tolist(), short_outputs.results):
                results[idx] = result

        if long_indices.numel() > 0:
            long_features, long_mask = self.__trim_to_valid_mel_frames(
                input_features[long_indices],
                attention_mask[long_indices],
            )
            long_outputs = self.__generate_features(
                long_features,
                long_mask,
                return_timestamps=True,
            )
            for idx, result in zip(long_indices.tolist(), long_outputs.results):
                results[idx] = result

        return SimpleNamespace(results=results)

    def __generate_features(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
        return_timestamps: bool,
    ) -> SimpleNamespace:
        num_return_sequences = 1 if self.__return_best else self.num_beams

        sequences = self.model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=self.num_beams,
            num_return_sequences=num_return_sequences,
            return_timestamps=return_timestamps,
        )

        if self.__return_best:
            return self.__greedy_format(sequences)
        else:
            return self.__beam_format(sequences, batch_size=input_features.size(0))

    def __preprocess(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
    ) -> SimpleNamespace:
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        waveforms_cpu = waveforms.detach().float().cpu()
        lengths_cpu = lengths.detach().cpu().tolist()

        audios = [
            waveforms_cpu[i, :int(length)].numpy()
            for i, length in enumerate(lengths_cpu)
        ]

        inputs = self.processor(
            audios,
            sampling_rate=DEFAULT_SAMPLE_RATE,
            return_tensors='pt',
            padding=True,
            return_attention_mask=True,
            truncation=False,
        )

        input_features = inputs.input_features.to(device=device, dtype=dtype)
        attention_mask = inputs.attention_mask.to(device=device)

        return SimpleNamespace(
            input_features=input_features,
            attention_mask=attention_mask,
        )

    def __greedy_format(
        self,
        sequences: torch.Tensor,
    ) -> SimpleNamespace:
        texts = self.processor.batch_decode(
            sequences,
            skip_special_tokens=True,
        )

        results = [
            [
                SimpleNamespace(
                    text=text,
                    score=0.0,
                    token_sequence=seq.tolist(),
                    timestamp=[],
                )
            ]
            for text, seq in zip(texts, sequences)
        ]

        return SimpleNamespace(
            results=results,
        )

    def __beam_format(
        self,
        sequences: torch.Tensor,
        batch_size: int,
    ) -> SimpleNamespace:
        texts = self.processor.batch_decode(
            sequences,
            skip_special_tokens=True,
        )

        num_hypotheses = self.num_beams

        results = []
        for b in range(batch_size):
            hypotheses = [
                SimpleNamespace(
                    text=texts[b * num_hypotheses + k],
                    score=0.0,
                    token_sequence=sequences[b * num_hypotheses + k].tolist(),
                    timestamp=[],
                )
                for k in range(num_hypotheses)
            ]
            results.append(hypotheses)

        return SimpleNamespace(
            results=results,
        )


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = WhisperLargeV3().to(device)
    model.eval()

    waveforms = torch.randn(3, 16000).to(device)
    lengths = torch.tensor([16000] * 3).to(device)

    model.greedy_mode()
    predictions = model(waveforms, lengths)
    print_log(predictions.results)

    model.beam_mode(num_beams=2, return_best_hypothesis=False)
    predictions = model(waveforms, lengths)
    print_log(predictions.results)
