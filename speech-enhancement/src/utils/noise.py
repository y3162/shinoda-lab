from typing import List
from types import SimpleNamespace
import torch
import torchaudio
import json
import duckdb as db
from src.config import DEFAULT_SAMPLE_RATE
from src.utils.print import print_warning


def get_noise_option(
    con: db.DuckDBPyConnection,
    noise_config_id: int,
) -> dict:
    rows = con.execute(
        """
        SELECT config_json FROM noise_configs WHERE id = ?
        """,
        [noise_config_id],
    ).fetchone()
    fetched_option = json.loads(rows[0])
    parsed_args = []
    for arg in fetched_option['args']:
        noise_id = arg.get('noise_id', None)
        if noise_id is not None:
            noise_path = con.execute(
                """
                SELECT audio_path FROM noises WHERE id = ?
                """,
                [noise_id],
            ).fetchone()
            arg['audiofile_path'] = noise_path[0]
            parsed_args.append(arg)
        else:
            parsed_args.append(arg)

    fetched_option['args'] = parsed_args

    return fetched_option


class NoiseGenerator:
    def __init__(
        self,
        settings: dict,
    ):
        self.settings = settings
        match settings['generator_type']:
            case 'additive':
                self.generator = AdditiveNoiseGenerator(settings)
            case _:
                raise NotImplementedError(f'Noise type {settings["type"]} not implemented')

    def generate(
        self,
        audio: torch.Tensor,
        sample_rate: int,
    )->torch.Tensor:
        return self.generator.generate(
            audio,
            sample_rate,
        )


class AdditiveNoiseGenerator:
    def __init__(
        self,
        settings: dict,
    ):
        self.settings = settings
        self.arg_settings = settings['args']

    def generate(
        self,
        audio: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        return add_noise(
            audio,
            sample_rate,
            self.arg_settings,
        )


def add_noise(
    audio: torch.Tensor,
    sample_rate: int,
    options: List[dict],
)->torch.Tensor:
    if sample_rate != DEFAULT_SAMPLE_RATE:
        print_warning(f'Sample rate must be DEFAULT_SAMPLE_RATE, but got {sample_rate}')

    noised_audio = audio
    used_options = []
    for option in options:
        match option['type']:
            case 'audiofile':
                result = audiofile_noise_generation(
                    audio=noised_audio,
                    sample_rate=sample_rate,
                    option=option,
                )
                noised_audio = result.audio
                used_options.append(result.option)
            case _:
                raise NotImplementedError(f'Noise type {option["type"]} not implemented')

    return SimpleNamespace(
        audio=noised_audio,
        options=used_options,
    )


def __parse_common_option(
    audio: torch.Tensor,
    sample_rate: int,
    option: dict,
)->SimpleNamespace:
    if sample_rate != DEFAULT_SAMPLE_RATE:
        print_warning(f'Sample rate must be DEFAULT_SAMPLE_RATE, but got {sample_rate}')
    snr_db = option['snr_db']

    start_frame = option.get('start_frame', None)
    end_frame = option.get('end_frame', None)
    start_ratio = option.get('start_ratio', None)
    end_ratio = option.get('end_ratio', None)

    if start_frame is not None:
        start_frame = max(start_frame, 0)
    elif start_ratio is not None:
        start_frame = int(audio.shape[-1] * start_ratio)
    else:
        start_frame = 0

    if end_frame is not None:
        end_frame = min(end_frame, audio.shape[-1])
    elif end_ratio is not None:
        end_frame = int(audio.shape[-1] * end_ratio)
    else:
        end_frame = audio.shape[-1]

    return SimpleNamespace(
        snr_db=snr_db,
        start_frame=start_frame,
        end_frame=end_frame,
    )


def audiofile_noise_generation(
    audio: torch.Tensor,
    sample_rate: int,
    option: dict={},
)->torch.Tensor:
    common_option = __parse_common_option(audio, sample_rate, option)
    snr_db = common_option.snr_db
    start_frame = common_option.start_frame
    end_frame = common_option.end_frame

    audiofile_path = option['audiofile_path']

    noise, sr = torchaudio.load(audiofile_path)
    if sr != DEFAULT_SAMPLE_RATE:
        noise = torchaudio.transforms.Resample(sr, DEFAULT_SAMPLE_RATE)(noise)
    if noise.dim() == 2:
        noise = noise.mean(dim=0)
    noise_duration = end_frame - start_frame
    if noise.shape[-1] < noise_duration:
        repeats = (noise_duration // noise.shape[-1]) + 1
        noise = noise.repeat(repeats)
    audiofile_start_frame = 0 if noise.shape[-1] == noise_duration else torch.randint(0, noise.shape[-1] - noise_duration, (1,)).item()
    audiofile_end_frame = audiofile_start_frame + noise_duration
    noise = noise[..., audiofile_start_frame:audiofile_end_frame]
    signal_power = audio.pow(2).mean()
    noise_power = noise.pow(2).mean()
    scale = torch.sqrt(
        signal_power / (10 ** (snr_db / 10) * noise_power)
    )
    noise = noise * scale
    audio[..., start_frame:end_frame] += noise

    option = {
        'type': 'audiofile',
        'snr_db': snr_db,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'audiofile_path': audiofile_path,
    }

    return SimpleNamespace(
        audio=audio,
        option=option,
    )
