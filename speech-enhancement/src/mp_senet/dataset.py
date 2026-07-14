import os
import random

import librosa
import torch


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


class VoiceBankPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        training_indexes,
        clean_wavs_dir,
        noisy_wavs_dir,
        segment_size,
        sampling_rate,
        split=True,
        shuffle=True,
    ):
        self.audio_indexes = training_indexes
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_indexes)
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

        clean_audio, noisy_audio = (
            torch.FloatTensor(clean_audio),
            torch.FloatTensor(noisy_audio),
        )
        norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
        clean_audio = (clean_audio * norm_factor).unsqueeze(0)
        noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)

        assert clean_audio.size(1) == noisy_audio.size(1)

        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                clean_audio = clean_audio[:, audio_start: audio_start + self.segment_size]
                noisy_audio = noisy_audio[:, audio_start: audio_start + self.segment_size]
            else:
                clean_audio = torch.nn.functional.pad(
                    clean_audio,
                    (0, self.segment_size - clean_audio.size(1)),
                    'constant',
                )
                noisy_audio = torch.nn.functional.pad(
                    noisy_audio,
                    (0, self.segment_size - noisy_audio.size(1)),
                    'constant',
                )

        return (clean_audio.squeeze(), noisy_audio.squeeze())

    def __len__(self):
        return len(self.audio_indexes)


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
            shuffle=False if h.num_gpus > 1 else True,
        )
        validset = VoiceBankPairDataset(
            validation_indexes,
            args.input_validation_clean_wavs_dir,
            args.input_validation_noisy_wavs_dir,
            h.segment_size,
            h.sampling_rate,
            split=False,
            shuffle=False,
        )
        return trainset, validset

    if args.dataset == 'librispeech':
        raise NotImplementedError(
            'LibriSpeech dataset is not implemented yet. '
            'Planned: DuckDB utterances + noise_configs with on-the-fly mixing.'
        )

    raise ValueError('Unsupported dataset: {}'.format(args.dataset))
