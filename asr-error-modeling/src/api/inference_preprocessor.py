from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from src.config import DEFAULT_SAMPLE_RATE

WHISPER_SHORT_FORM_MEL_FRAMES = 3000


@dataclass
class DataLoaderDefaults:
    batch_size: int
    num_workers: int
    max_batch_frames: int | None = None
    max_batch_size: int | None = None
    prefetch_factor: int = 4


class FrameBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        frame_counts: List[int],
        max_total_frames: int,
        max_batch_size: int | None = None,
    ):
        self.batches: List[List[int]] = []
        batch: List[int] = []
        batch_frames = 0
        for idx, frames in enumerate(frame_counts):
            over_frame_limit = batch and batch_frames + frames > max_total_frames
            over_size_limit = max_batch_size is not None and len(batch) >= max_batch_size
            if over_frame_limit or over_size_limit:
                self.batches.append(batch)
                batch = []
                batch_frames = 0
            batch.append(idx)
            batch_frames += frames
        if batch:
            self.batches.append(batch)

    def __iter__(
        self,
    ) -> Iterator[List[int]]:
        yield from self.batches

    def __len__(
        self,
    ) -> int:
        return len(self.batches)


def _dataloader_worker_init_fn(
    _worker_id: int,
) -> None:
    torch.set_num_threads(1)


class InferencePreprocessor(ABC):
    def collate(
        self,
        batch: List[Tuple[int, torch.Tensor]],
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        utterance_ids = [utterance_id for utterance_id, _ in batch]
        max_audio_len = max(audio.shape[0] for _, audio in batch)
        lengths = [audio.shape[0] for _, audio in batch]
        padded_batch = []
        for _, audio in batch:
            padding_len = max_audio_len - audio.shape[0]
            padded_batch.append(F.pad(audio, (0, padding_len)))
        return utterance_ids, torch.stack(padded_batch), torch.tensor(lengths)

    def transfer_batch(
        self,
        batch: tuple,
        device: torch.device,
    ) -> tuple:
        utterance_ids, audio, lengths = batch
        non_blocking = device.type == 'cuda'
        return (
            utterance_ids,
            audio.to(device, non_blocking=non_blocking),
            lengths.to(device, non_blocking=non_blocking),
        )

    @abstractmethod
    def infer_device(
        self,
        model: nn.Module,
        batch: tuple,
        device: torch.device,
    ) -> object:
        pass

    def dataloader_defaults(
        self,
    ) -> DataLoaderDefaults:
        return DataLoaderDefaults(batch_size=128, num_workers=8)

    def create_dataloader(
        self,
        dataset: Dataset,
        *,
        batch_size: int | None = None,
        num_workers: int | None = None,
        max_batch_frames: int | None = None,
        prefetch_factor: int | None = None,
        use_cuda: bool = False,
    ) -> DataLoader:
        defaults = self.dataloader_defaults()
        _num_workers = num_workers if num_workers is not None else defaults.num_workers
        _prefetch_factor = prefetch_factor if prefetch_factor is not None else defaults.prefetch_factor
        _max_batch_frames = max_batch_frames if max_batch_frames is not None else defaults.max_batch_frames
        _max_batch_size = batch_size if batch_size is not None else defaults.max_batch_size
        _batch_size = batch_size if batch_size is not None else defaults.batch_size

        kwargs: dict = dict(
            collate_fn=self.collate,
            num_workers=_num_workers,
            pin_memory=use_cuda,
            worker_init_fn=_dataloader_worker_init_fn,
        )
        if _num_workers > 0:
            kwargs['prefetch_factor'] = _prefetch_factor
            kwargs['persistent_workers'] = True
        if _max_batch_frames is not None and hasattr(dataset, 'frame_counts'):
            kwargs['batch_sampler'] = FrameBatchSampler(
                dataset.frame_counts,
                _max_batch_frames,
                _max_batch_size,
            )
        else:
            kwargs['batch_size'] = _batch_size
            kwargs['shuffle'] = False
        return DataLoader(dataset, **kwargs)


class InferencePipeline:
    def __init__(
        self,
        preprocessor: InferencePreprocessor,
        model: nn.Module,
        device: torch.device,
    ):
        self.preprocessor = preprocessor
        self.model = model
        self.device = device

    def run(
        self,
        dataloader: DataLoader,
    ) -> Iterator[Tuple[tuple, object]]:
        for batch in dataloader:
            device_batch = self.preprocessor.transfer_batch(batch, self.device)
            outputs = self.preprocessor.infer_device(self.model, device_batch, self.device)
            yield batch, outputs


class WaveformInferencePreprocessor(InferencePreprocessor):
    def infer_device(
        self,
        model: nn.Module,
        batch: tuple,
        device: torch.device,
    ) -> object:
        _, audio, lengths = batch
        return model(audio, lengths)


class WhisperInferencePreprocessor(InferencePreprocessor):
    MAX_BATCH_FRAMES = DEFAULT_SAMPLE_RATE * 30 * 64
    MAX_BATCH_SIZE = 64

    def __init__(
        self,
        processor,
    ):
        self.processor = processor

    def transfer_batch(
        self,
        batch: tuple,
        device: torch.device,
    ) -> tuple:
        utterance_ids, waveforms, lengths = batch
        audios = [
            waveforms[i, :int(lengths[i])].numpy()
            for i in range(waveforms.shape[0])
        ]
        inputs = self.processor(
            audios,
            sampling_rate=DEFAULT_SAMPLE_RATE,
            return_tensors='pt',
            padding=True,
            return_attention_mask=True,
            truncation=False,
        )
        dtype = torch.float16 if device.type == 'cuda' else torch.float32
        non_blocking = device.type == 'cuda'
        return (
            utterance_ids,
            inputs.input_features.to(device=device, dtype=dtype, non_blocking=non_blocking),
            inputs.attention_mask.to(device=device, non_blocking=non_blocking),
        )

    def infer_device(
        self,
        model: nn.Module,
        batch: tuple,
        device: torch.device,
    ) -> object:
        _, input_features, attention_mask = batch
        return model.forward_from_features(input_features, attention_mask)

    def dataloader_defaults(
        self,
    ) -> DataLoaderDefaults:
        return DataLoaderDefaults(
            batch_size=self.MAX_BATCH_SIZE,
            num_workers=12,
            max_batch_frames=self.MAX_BATCH_FRAMES,
            max_batch_size=self.MAX_BATCH_SIZE,
            prefetch_factor=8,
        )
