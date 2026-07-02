import argparse

import torchaudio
from tqdm import tqdm

from src.utils.demand import (
    find_all_audio_files,
    get_noise_type,
)
from src.config import (
    DEMAND_CLIPPED_ROOT,
)
from src.utils.print import (
    print_error,
    print_log,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=20)
    args = parser.parse_args()

    DURATION_SEC = args.duration

    if DEMAND_CLIPPED_ROOT.exists():
        print_error(f'DEMAND_CLIPPED_ROOT {DEMAND_CLIPPED_ROOT} already exists')
        return
    else:
        print_log(f'Creating DEMAND_CLIPPED_ROOT {DEMAND_CLIPPED_ROOT}')
        print_log(f'DURATION_SEC: {DURATION_SEC} seconds')
        DEMAND_CLIPPED_ROOT.mkdir(parents=True, exist_ok=True)

    for audio_file in tqdm(find_all_audio_files(), desc='Clipping audio files'):
        if audio_file.stem != 'ch01':
            continue
        audio, sr = torchaudio.load(audio_file)
        num_clips = audio.shape[-1] // (sr * DURATION_SEC)
        for i in tqdm(range(num_clips), desc='Clipping audio file', leave=False):
            start_frame = i * (sr * DURATION_SEC)
            end_frame = start_frame + (sr * DURATION_SEC)
            clip = audio[..., start_frame:end_frame]
            clip_path = DEMAND_CLIPPED_ROOT / get_noise_type(audio_file) / f"{audio_file.stem}_{DURATION_SEC * i}-{DURATION_SEC * (i + 1)}.wav"
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(clip_path, clip, sr)


if __name__ == '__main__':
    main()
