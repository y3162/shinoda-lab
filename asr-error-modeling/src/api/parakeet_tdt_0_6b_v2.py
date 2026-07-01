from typing import List
from types import SimpleNamespace

import torch
import torch.nn as nn
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.utils import logging as nemo_logging
nemo_logging.set_verbosity(nemo_logging.ERROR)

from src.api.inference_preprocessor import WaveformInferencePreprocessor
from src.utils.print import print_log


class ParakeetTDT06BV2(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name='nvidia/parakeet-tdt-0.6b-v2')

    @property
    def __return_best(
        self,
    ) -> bool:
        is_greedy = self.model.cfg.decoding.strategy == 'greedy_batch'
        is_beam_best = self.model.cfg.decoding.strategy == 'beam' and self.model.cfg.decoding.beam.return_best_hypothesis
        return is_greedy or is_beam_best

    def greedy_mode(
        self,
    ):
        decoding_cfg = self.model.cfg.decoding
        decoding_cfg.strategy = 'greedy_batch'
        self.model.change_decoding_strategy(decoding_cfg)

    def beam_mode(
        self,
        num_beams: int,
        strategy: str = 'malsd_batch',
        return_best_hypothesis: bool = False,
    ):
        decoding_cfg = self.model.cfg.decoding
        decoding_cfg.strategy = strategy
        decoding_cfg.beam.beam_size = num_beams
        decoding_cfg.beam.return_best_hypothesis = return_best_hypothesis
        self.model.change_decoding_strategy(decoding_cfg)

    def create_inference_preprocessor(
        self,
    ) -> WaveformInferencePreprocessor:
        return WaveformInferencePreprocessor()

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
    ) -> SimpleNamespace:

        processed_signal, processed_len = self.model.preprocessor(
            input_signal=waveforms,
            length=lengths,
        )

        encoded, encoded_len = self.model.encoder(
            audio_signal=processed_signal,
            length=processed_len,
        )

        predictions = self.model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded,
            encoded_lengths=encoded_len,
            return_hypotheses=True,
        )

        if self.__return_best:
            return self.__greedy_format(predictions)
        else:
            return self.__beam_format(predictions)

    def __greedy_format(
        self,
        predictions: List[Hypothesis],
    ) -> SimpleNamespace:
        results = [
            [SimpleNamespace(
                text=p.text,
                score=p.score,
                token_sequence=p.y_sequence.tolist(),
                timestamp=p.timestamp.tolist(),
            )] for p in predictions]

        return SimpleNamespace(
            results=results,
        )

    def __beam_format(
        self,
        predictions: List[List[Hypothesis]],
    ) -> SimpleNamespace:
        results = [
            [SimpleNamespace(
                text=h.text,
                score=h.score,
                token_sequence=h.y_sequence.tolist(),
                timestamp=h.timestamp.tolist(),
            ) for h in sorted(p, key=lambda x: x.score, reverse=True)] for p in predictions]

        return SimpleNamespace(
            results=results,
        )


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ParakeetTDT06BV2().to(device)
    model.eval()

    waveforms = torch.randn(3, 16000).to(device)
    lengths = torch.tensor([16000]*3).to(device)

    model.greedy_mode()
    predictions = model(waveforms, lengths)
    print_log(predictions.results)

    model.beam_mode(num_beams=2, return_best_hypothesis=False)
    predictions = model(waveforms, lengths)
    print_log(predictions.results)
