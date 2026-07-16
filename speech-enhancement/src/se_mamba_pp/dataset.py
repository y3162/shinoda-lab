import os
import random
from pathlib import Path

import duckdb as db
import librosa
import torch
import torchaudio

from src.utils.noise import NoiseGenerator, get_noise_option


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
    clean_audio, noisy_audio = (
        torch.FloatTensor(clean_audio),
        torch.FloatTensor(noisy_audio),
    )
    norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
    clean_audio = (clean_audio * norm_factor).unsqueeze(0)
    noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)

    assert clean_audio.size(1) == noisy_audio.size(1)

    if split:
        if clean_audio.size(1) >= segment_size:
            max_audio_start = clean_audio.size(1) - segment_size
            audio_start = random.randint(0, max_audio_start)
            clean_audio = clean_audio[:, audio_start: audio_start + segment_size]
            noisy_audio = noisy_audio[:, audio_start: audio_start + segment_size]
        else:
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

    return clean_audio.squeeze(), noisy_audio.squeeze()


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

    def __getitem__(self, index):
        filename = self.audio_indexes[index]
        clean_audio, _ = librosa.load(
            os.path.join(self.clean_wavs_dir, filename + '.wav'),
            sr=self.sampling_rate,
        )
        noisy_audio, _ = librosa.load(
            os.path.join(self.noisy_wavs_dir, filename + '.wav'),
            sr=self.sampling_rate,
        )
        length = min(len(clean_audio), len(noisy_audio))
        clean_audio, noisy_audio = clean_audio[:length], noisy_audio[:length]

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


def build_datasets(args, h):
    if args.dataset == 'voicebank':
        training_indexes, validation_indexes = get_voicebank_filelist(
            args.input_training_file,
            args.input_validation_file,
        )
        trainset = VoiceBankPairDataset(
            training_indexes,
            args.input_clean_wavs_dir,
            args.input_noisy_wavs_dir,
            h.segment_size,
            h.sampling_rate,
            split=True,
        )
        validset = VoiceBankPairDataset(
            validation_indexes,
            args.input_validation_clean_wavs_dir,
            args.input_validation_noisy_wavs_dir,
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
