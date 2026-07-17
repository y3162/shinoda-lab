import os
import random
from pathlib import Path

import duckdb as db
import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.utils.noise import NoiseGenerator, get_noise_option


def _load_wav_mono(path, sampling_rate, frame_offset=0, num_frames=-1):
    if num_frames is None or num_frames < 0:
        waveform, sample_rate = sf.read(path, dtype='float32', always_2d=True)
    else:
        waveform, sample_rate = sf.read(
            path,
            start=frame_offset,
            stop=frame_offset + num_frames,
            dtype='float32',
            always_2d=True,
        )

    waveform = waveform.mean(axis=1) if waveform.shape[1] > 1 else waveform[:, 0]
    waveform = torch.from_numpy(np.ascontiguousarray(waveform))

    if sample_rate != sampling_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            sampling_rate,
        )

    return waveform


def get_voicebank_filelist(training_file, validation_file):
    with open(training_file, 'r', encoding='utf-8') as fi:
        training_indexes = [
            x.split('|')[0].split()[0]
            for x in fi.read().split('\n')
            if len(x) > 0
        ]

    with open(validation_file, 'r', encoding='utf-8') as fi:
        validation_indexes = [
            x.split('|')[0].split()[0]
            for x in fi.read().split('\n')
            if len(x) > 0
        ]

    return training_indexes, validation_indexes


def normalize_and_segment(clean_audio, noisy_audio, segment_size, split):
    if not torch.is_tensor(clean_audio):
        clean_audio = torch.as_tensor(clean_audio, dtype=torch.float32)
    else:
        clean_audio = clean_audio.float()
    if not torch.is_tensor(noisy_audio):
        noisy_audio = torch.as_tensor(noisy_audio, dtype=torch.float32)
    else:
        noisy_audio = noisy_audio.float()

    clean_audio = clean_audio.reshape(-1)
    noisy_audio = noisy_audio.reshape(-1)
    length = min(clean_audio.numel(), noisy_audio.numel())
    clean_audio = clean_audio[:length]
    noisy_audio = noisy_audio[:length]

    norm_factor = torch.sqrt(length / torch.sum(noisy_audio ** 2.0).clamp_min(1e-12))
    clean_audio = (clean_audio * norm_factor).unsqueeze(0)
    noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)

    if split:
        if clean_audio.size(1) > segment_size:
            max_audio_start = clean_audio.size(1) - segment_size
            audio_start = random.randint(0, max_audio_start)
            clean_audio = clean_audio[:, audio_start: audio_start + segment_size]
            noisy_audio = noisy_audio[:, audio_start: audio_start + segment_size]
        elif clean_audio.size(1) < segment_size:
            clean_audio = torch.nn.functional.pad(
                clean_audio,
                (0, segment_size - clean_audio.size(1)),
                'constant',
            )
            noisy_audio = torch.nn.functional.pad(
                noisy_audio,
                (0, segment_size - noisy_audio.size(1)),
                'constant',
            )

    return clean_audio.squeeze(0), noisy_audio.squeeze(0)


class VoiceBankPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        training_indexes,
        clean_wavs_dir,
        noisy_wavs_dir,
        segment_size,
        sampling_rate,
        split=True,
    ):
        self.audio_indexes = training_indexes
        self.clean_wavs_dir = clean_wavs_dir
        self.noisy_wavs_dir = noisy_wavs_dir
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.num_frames = []
        for filename in training_indexes:
            clean_info = sf.info(os.path.join(clean_wavs_dir, filename + '.wav'))
            noisy_info = sf.info(os.path.join(noisy_wavs_dir, filename + '.wav'))
            self.num_frames.append(min(clean_info.frames, noisy_info.frames))

    def __getitem__(self, index):
        filename = self.audio_indexes[index]
        clean_path = os.path.join(self.clean_wavs_dir, filename + '.wav')
        noisy_path = os.path.join(self.noisy_wavs_dir, filename + '.wav')
        num_frames = self.num_frames[index]

        if self.split and num_frames >= self.segment_size:
            frame_offset = random.randint(0, num_frames - self.segment_size)
            clean_audio = _load_wav_mono(
                clean_path,
                self.sampling_rate,
                frame_offset=frame_offset,
                num_frames=self.segment_size,
            )
            noisy_audio = _load_wav_mono(
                noisy_path,
                self.sampling_rate,
                frame_offset=frame_offset,
                num_frames=self.segment_size,
            )
            return normalize_and_segment(
                clean_audio,
                noisy_audio,
                self.segment_size,
                split=False,
            )

        clean_audio = _load_wav_mono(clean_path, self.sampling_rate)
        noisy_audio = _load_wav_mono(noisy_path, self.sampling_rate)
        return normalize_and_segment(
            clean_audio,
            noisy_audio,
            self.segment_size,
            self.split,
        )

    def __len__(self):
        return len(self.audio_indexes)


def _load_librispeech_utterances(con, splits):
    placeholders = ', '.join(['?'] * len(splits))
    rows = con.execute(
        f"""
        SELECT id, audio_path
        FROM utterances
        WHERE split IN ({placeholders})
        ORDER BY id
        """,
        list(splits),
    ).fetchall()
    if not rows:
        raise ValueError(
            'No utterances found for splits: {}'.format(', '.join(splits))
        )
    return rows


def _max_frame_count(con, splits):
    placeholders = ', '.join(['?'] * len(splits))
    row = con.execute(
        f"""
        SELECT MAX(frame_count)
        FROM utterances
        WHERE split IN ({placeholders})
        """,
        list(splits),
    ).fetchone()
    if row is None or row[0] is None:
        raise ValueError(
            'No frame_count found for splits: {}'.format(', '.join(splits))
        )
    return int(row[0])


def _load_noise_options(con, noise_config_ids):
    if noise_config_ids is None:
        rows = con.execute(
            """
            SELECT id
            FROM noise_configs
            WHERE json_array_length(config_json->'$.args') <> 0
            ORDER BY id
            """
        ).fetchall()
        noise_config_ids = [row[0] for row in rows]
    else:
        noise_config_ids = list(noise_config_ids)

    if not noise_config_ids:
        raise ValueError('No noise_config_ids available for LibriSpeechNoiseDataset')

    options = []
    for noise_config_id in noise_config_ids:
        option = get_noise_option(con, noise_config_id)
        if not option.get('args'):
            continue
        options.append(option)

    if not options:
        raise ValueError('Resolved noise options are empty (clean-only configs?)')

    return options


class LibriSpeechNoiseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sql_root,
        splits,
        segment_size,
        sampling_rate,
        noise_config_ids=None,
        split=True,
        max_frames=None,
    ):
        if not splits:
            raise ValueError('splits must be a non-empty list')

        con = db.connect(str(sql_root), read_only=True)
        try:
            self.utterances = _load_librispeech_utterances(con, splits)
            self.noise_options = _load_noise_options(con, noise_config_ids)
        finally:
            con.close()

        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.max_frames = max_frames

    def _load_clean_audio(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        if sample_rate != self.sampling_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                self.sampling_rate,
            )

        if self.max_frames is not None and waveform.shape[-1] > self.max_frames:
            waveform = waveform[: self.max_frames]

        return waveform

    def __getitem__(self, index):
        _, audio_path = self.utterances[index]
        clean_audio = self._load_clean_audio(audio_path)

        noise_option = random.choice(self.noise_options)
        generator = NoiseGenerator(noise_option)
        noisy_result = generator.generate(
            clean_audio.clone(),
            self.sampling_rate,
        )
        noisy_audio = noisy_result.audio

        length = min(clean_audio.shape[-1], noisy_audio.shape[-1])
        clean_audio = clean_audio[:length]
        noisy_audio = noisy_audio[:length]

        return normalize_and_segment(
            clean_audio,
            noisy_audio,
            self.segment_size,
            self.split,
        )

    def __len__(self):
        return len(self.utterances)


def _count_wavs(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    return sum(1 for _ in directory.glob('*.wav'))


def _voicebank_audio_dir(raw_dir, sampling_rate):
    raw_dir = Path(raw_dir)
    raw_count = _count_wavs(raw_dir)
    if raw_count == 0:
        return str(raw_dir)

    candidates = []
    env_root = os.environ.get('VOICEBANK_16K_ROOT')
    if env_root:
        candidates.append(Path(env_root) / raw_dir.name)
    candidates.append(Path('/tmp/seki') / 'VoiceBank+DEMAND_{}'.format(sampling_rate) / raw_dir.name)
    candidates.append(
        Path('data')
        / 'processed'
        / 'VoiceBank+DEMAND_{}'.format(sampling_rate)
        / raw_dir.name
    )

    for cache_dir in candidates:
        if _count_wavs(cache_dir) >= raw_count:
            return str(cache_dir)
    return str(raw_dir)


def build_datasets(args, h):
    if args.dataset == 'voicebank':
        training_indexes, validation_indexes = get_voicebank_filelist(
            args.input_training_file,
            args.input_validation_file,
        )
        train_clean_dir = _voicebank_audio_dir(
            args.input_clean_wavs_dir,
            h.sampling_rate,
        )
        train_noisy_dir = _voicebank_audio_dir(
            args.input_noisy_wavs_dir,
            h.sampling_rate,
        )
        valid_clean_dir = _voicebank_audio_dir(
            args.input_validation_clean_wavs_dir,
            h.sampling_rate,
        )
        valid_noisy_dir = _voicebank_audio_dir(
            args.input_validation_noisy_wavs_dir,
            h.sampling_rate,
        )
        trainset = VoiceBankPairDataset(
            training_indexes,
            train_clean_dir,
            train_noisy_dir,
            h.segment_size,
            h.sampling_rate,
            split=True,
        )
        validset = VoiceBankPairDataset(
            validation_indexes,
            valid_clean_dir,
            valid_noisy_dir,
            h.segment_size,
            h.sampling_rate,
            split=False,
        )
        return trainset, validset

    if args.dataset == 'librispeech':
        if not args.train_splits:
            raise ValueError('--train_splits is required for --dataset librispeech')
        if not args.validation_splits:
            raise ValueError(
                '--validation_splits is required for --dataset librispeech'
            )
        if args.sql_root is None:
            raise ValueError('--sql_root is required for --dataset librispeech')

        sql_root = Path(args.sql_root)
        con = db.connect(str(sql_root), read_only=True)
        max_frames = _max_frame_count(con, args.train_splits)
        con.close()

        trainset = LibriSpeechNoiseDataset(
            sql_root,
            args.train_splits,
            h.segment_size,
            h.sampling_rate,
            noise_config_ids=args.noise_config_ids,
            split=True,
        )
        validset = LibriSpeechNoiseDataset(
            sql_root,
            args.validation_splits,
            h.segment_size,
            h.sampling_rate,
            noise_config_ids=args.noise_config_ids,
            split=False,
            max_frames=max_frames,
        )
        return trainset, validset

    raise ValueError('Unsupported dataset: {}'.format(args.dataset))
