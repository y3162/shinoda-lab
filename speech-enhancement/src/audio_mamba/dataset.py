import random

import duckdb as db
import torch
import torchaudio

from src.config import resolve_project_path

CLEAN_NOISE_TYPE = 'clean'


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


def _infer_noise_split(utterance_splits):
    kinds = set()
    for name in utterance_splits:
        if name.startswith('train'):
            kinds.add('train')
        elif name.startswith('dev'):
            kinds.add('dev')
        elif name.startswith('test'):
            kinds.add('test')
        else:
            raise ValueError(f'Cannot map utterance split to noise split: {name}')
    if len(kinds) != 1:
        raise ValueError(
            f'Utterance splits must map to one noise split, got {utterance_splits} -> {kinds}'
        )
    return next(iter(kinds))


def _build_noise_type_index(con):
    rows = con.execute(
        """
        SELECT DISTINCT noise_type
        FROM noises
        ORDER BY noise_type
        """,
    ).fetchall()
    noise_types = [row[0] for row in rows]
    # clean + DEMAND types (sorted). clean is always index 0.
    return {
        CLEAN_NOISE_TYPE: 0,
        **{name: index + 1 for index, name in enumerate(noise_types)},
    }


def _resolve_noise_type(con, noise_id):
    row = con.execute(
        """
        SELECT noise_type
        FROM noises
        WHERE id = ?
        """,
        [int(noise_id)],
    ).fetchone()
    if row is None:
        raise ValueError(f'noise_id {noise_id} not found in noises table')
    return row[0]


def _make_clean_option(noise_type_to_idx):
    return {
        'generator_type': 'additive',
        'kind': 'clean',
        'args': [],
        'noise_type': CLEAN_NOISE_TYPE,
        'noise_type_idx': noise_type_to_idx[CLEAN_NOISE_TYPE],
        'is_clean': True,
    }


def _load_noise_options(con, noise_config_ids, noise_split=None):
    from src.utils.noise import get_noise_option

    noise_type_to_idx = _build_noise_type_index(con)
    options = []

    if noise_config_ids is None:
        if noise_split is None:
            raise ValueError('noise_split is required when noise_config_ids is None')
        single_rows = con.execute(
            """
            SELECT id
            FROM noise_configs
            WHERE json_array_length(config_json->'$.args') = 1
              AND json_extract_string(config_json, '$.kind') = 'single'
              AND json_extract_string(config_json, '$.split') = ?
            ORDER BY id
            """,
            [noise_split],
        ).fetchall()
        noise_config_ids = [row[0] for row in single_rows]
        # Always include a clean class option for both train/dev pools.
        options.append(_make_clean_option(noise_type_to_idx))
    else:
        noise_config_ids = list(noise_config_ids)

    if not noise_config_ids and not options:
        raise ValueError('No noise config ids available')

    for noise_config_id in noise_config_ids:
        option = get_noise_option(con, noise_config_id)
        args = option.get('args') or []
        if len(args) == 0:
            clean_option = _make_clean_option(noise_type_to_idx)
            if not any(item.get('is_clean') for item in options):
                options.append(clean_option)
            continue
        if option.get('kind') not in (None, 'single'):
            continue
        if len(args) != 1:
            continue
        if noise_split is not None and option.get('split') not in (None, noise_split):
            continue

        noise_id = args[0].get('noise_id')
        if noise_id is None:
            continue
        noise_type = _resolve_noise_type(con, noise_id)
        if noise_type not in noise_type_to_idx:
            raise ValueError(f'Unknown noise_type {noise_type} for noise_id {noise_id}')

        option = dict(option)
        option['noise_type'] = noise_type
        option['noise_type_idx'] = noise_type_to_idx[noise_type]
        option['is_clean'] = False
        options.append(option)

    if not options:
        raise ValueError('Resolved noise options are empty')

    options_by_type = {name: [] for name in noise_type_to_idx}
    for option in options:
        options_by_type[option['noise_type']].append(option)

    for name, type_options in options_by_type.items():
        if not type_options:
            raise ValueError(f'No noise options for class {name}')

    # Index order: clean at 0, then DEMAND types by sorted name.
    class_names = [
        name for name, _ in sorted(
            noise_type_to_idx.items(),
            key=lambda item: item[1],
        )
    ]
    return options, noise_type_to_idx, options_by_type, class_names


def normalize_and_segment(waveform, segment_size, split):
    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    norm_factor = torch.sqrt(len(waveform) / torch.sum(waveform ** 2.0))
    waveform = waveform * norm_factor

    if split:
        if waveform.size(0) >= segment_size:
            max_audio_start = waveform.size(0) - segment_size
            audio_start = random.randint(0, max_audio_start)
            waveform = waveform[audio_start: audio_start + segment_size]
        else:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, segment_size - waveform.size(0)),
                'constant',
            )

    return waveform


class MelTransform(torch.nn.Module):
    def __init__(self, mel_cfg, sampling_rate, target_time_frames, deterministic=False):
        super().__init__()
        self.eps = float(mel_cfg.eps)
        self.target_time_frames = int(target_time_frames)
        self.deterministic = deterministic
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=mel_cfg.n_fft,
            win_length=mel_cfg.win_size,
            hop_length=mel_cfg.hop_size,
            n_mels=mel_cfg.n_mels,
            f_min=mel_cfg.f_min,
            f_max=mel_cfg.f_max,
        )

    def forward(self, waveform):
        mel = self.mel(waveform.unsqueeze(0))
        mel = torch.log(mel + self.eps)
        mel = self._fit_time(mel)
        return mel

    def _fit_time(self, mel):
        _, _, time_frames = mel.shape
        target = self.target_time_frames
        if time_frames > target:
            if self.deterministic:
                start = (time_frames - target) // 2
            else:
                start = random.randint(0, time_frames - target)
            mel = mel[:, :, start: start + target]
        elif time_frames < target:
            pad = target - time_frames
            mel = torch.nn.functional.pad(mel, (0, pad), 'constant')
        return mel


class SpecAugment(torch.nn.Module):
    def __init__(self, time_mask, freq_mask):
        super().__init__()
        self.time_mask = int(time_mask)
        self.freq_mask = int(freq_mask)

    def forward(self, mel):
        mel = mel.clone()
        _, freq_bins, time_frames = mel.shape

        if self.freq_mask > 0 and freq_bins > 1:
            f_max = min(self.freq_mask, freq_bins - 1)
            f = random.randint(0, f_max)
            f0 = random.randint(0, freq_bins - f)
            mel[:, f0: f0 + f, :] = 0.0

        if self.time_mask > 0 and time_frames > 1:
            t_max = min(self.time_mask, time_frames - 1)
            t = random.randint(0, t_max)
            t0 = random.randint(0, time_frames - t)
            mel[:, :, t0: t0 + t] = 0.0

        return mel


class LibriSpeechNoiseEstimationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sql_root,
        splits,
        data_cfg,
        model_cfg,
        noise_config_ids=None,
        noise_split=None,
        split=True,
        max_frames=None,
        seed=None,
    ):
        if not splits:
            raise ValueError('splits must be a non-empty list')

        if noise_split is None:
            noise_split = _infer_noise_split(splits)

        con = db.connect(str(sql_root), read_only=True)
        try:
            self.utterances = _load_librispeech_utterances(con, splits)
            (
                self.noise_options,
                self.noise_type_to_idx,
                self.options_by_type,
                self.class_names,
            ) = _load_noise_options(
                con,
                noise_config_ids,
                noise_split=noise_split,
            )
        finally:
            con.close()

        num_classes = int(model_cfg.num_classes)
        if len(self.noise_type_to_idx) != num_classes:
            raise ValueError(
                f'Expected {num_classes} noise classes, got {len(self.noise_type_to_idx)}: '
                f'{sorted(self.noise_type_to_idx)}'
            )
        if len(self.class_names) != num_classes:
            raise ValueError(
                f'Expected {num_classes} class_names, got {len(self.class_names)}'
            )
        self.clean_class_index = self.noise_type_to_idx[CLEAN_NOISE_TYPE]

        self.segment_size = int(data_cfg.segment_size)
        self.sampling_rate = int(data_cfg.sampling_rate)
        self.split = split
        self.max_frames = max_frames
        self.seed = seed
        self.target_time_frames = int(model_cfg.spectrogram_size[1])

        self.mel_transform = MelTransform(
            data_cfg.mel,
            self.sampling_rate,
            self.target_time_frames,
            deterministic=not split,
        )
        self.specaug = None
        if split and hasattr(data_cfg, 'specaug'):
            self.specaug = SpecAugment(
                data_cfg.specaug.time_mask,
                data_cfg.specaug.freq_mask,
            )

    def _load_clean_audio(self, audio_path):
        waveform, sample_rate = torchaudio.load(resolve_project_path(audio_path))
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

    def _sample_noise_option(self, rng):
        class_name = rng.choice(self.class_names)
        return rng.choice(self.options_by_type[class_name])

    def __getitem__(self, index):
        from src.utils.noise import NoiseGenerator
        from src.utils.validation import deterministic_noise_seed

        _, audio_path = self.utterances[index]
        clean_audio = self._load_clean_audio(audio_path)

        if self.seed is None:
            noise_option = self._sample_noise_option(random)
            torch_rng = None
        else:
            sample_seed = deterministic_noise_seed(self.seed, index)
            py_rng = random.Random(sample_seed)
            noise_option = self._sample_noise_option(py_rng)
            torch_rng = torch.Generator()
            torch_rng.manual_seed(sample_seed & 0xFFFFFFFFFFFFFFFF)

        generator = NoiseGenerator(noise_option)
        noisy_result = generator.generate(
            clean_audio.clone(),
            self.sampling_rate,
            rng=torch_rng,
        )
        noisy_audio = noisy_result.audio

        length = min(clean_audio.shape[-1], noisy_audio.shape[-1])
        noisy_audio = noisy_audio[:length]

        noisy_audio = normalize_and_segment(
            noisy_audio,
            self.segment_size,
            self.split,
        )
        mel = self.mel_transform(noisy_audio)
        if self.specaug is not None:
            mel = self.specaug(mel)

        noise_type_idx = torch.tensor(
            noise_option['noise_type_idx'],
            dtype=torch.long,
        )
        is_clean = bool(noise_option.get('is_clean')) or not noisy_result.options
        if is_clean:
            snr_db = torch.tensor(0.0, dtype=torch.float32)
            snr_valid = torch.tensor(False)
        else:
            snr_db = torch.tensor(
                float(noisy_result.options[0]['snr_db']),
                dtype=torch.float32,
            )
            snr_valid = torch.tensor(True)
        return mel, noise_type_idx, snr_db, snr_valid

    def __len__(self):
        return len(self.utterances)


def build_datasets(config):
    data = config.data
    if data.dataset != 'librispeech':
        raise ValueError(f'Unsupported dataset: {data.dataset}')

    ls = data.librispeech
    if not ls.train_splits:
        raise ValueError('data.librispeech.train_splits is required')
    if not ls.validation_splits:
        raise ValueError('data.librispeech.validation_splits is required')
    if ls.sql_root is None:
        raise ValueError('data.librispeech.sql_root is required')

    sql_root = resolve_project_path(ls.sql_root)
    con = db.connect(str(sql_root), read_only=True)
    max_frames = _max_frame_count(con, ls.train_splits)
    con.close()

    trainset = LibriSpeechNoiseEstimationDataset(
        sql_root,
        ls.train_splits,
        data,
        config.model,
        noise_config_ids=ls.noise_config_ids,
        noise_split=getattr(ls, 'noise_split', None) or 'train',
        split=True,
    )
    validset = LibriSpeechNoiseEstimationDataset(
        sql_root,
        ls.validation_splits,
        data,
        config.model,
        noise_config_ids=ls.noise_config_ids,
        noise_split='dev',
        split=False,
        max_frames=max_frames,
        seed=config.train.env.seed,
    )
    return trainset, validset
