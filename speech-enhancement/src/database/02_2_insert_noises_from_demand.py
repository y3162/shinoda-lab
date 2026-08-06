from typing import Iterable, Iterator, List, Tuple

NoiseRow = Tuple[str, str, str, str, int, int, int]

import duckdb as db
from tqdm import tqdm
import torchaudio

from src.utils.print import (
    print_log,
    print_error,
)
from src.utils.demand import (
    find_clipped_audio_files,
    get_noise_type,
    parse_clip_split,
)
from src.config import PROJECT_ROOT, SQL_ROOT


def batched(
    iterable: Iterable[NoiseRow],
    batch_size: int,
) -> Iterator[List[NoiseRow]]:
    batch = []

    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def iter_noise_rows() -> Iterator[NoiseRow]:
    audio_files = sorted(find_clipped_audio_files())

    for audio_file in tqdm(
        audio_files,
        desc='Parsing clipped DEMAND files',
        unit='file',
    ):
        split = parse_clip_split(audio_file)
        if split is None:
            continue
        audio, sample_rate = torchaudio.load(str(audio_file))
        assert audio.dim() == 2, 'Audio must have 2 dimensions'
        channels = audio.shape[0]
        frame_count = audio.shape[1]
        yield (
            'demand',
            str(audio_file.relative_to(PROJECT_ROOT)),
            get_noise_type(audio_file),
            split,
            sample_rate,
            channels,
            frame_count,
        )


def insert_noises(
    con: db.DuckDBPyConnection,
    batch_size: int = 1_000,
) -> None:
    insert_sql = """
        INSERT INTO noises (
            dataset_name,
            audio_path,
            noise_type,
            split,
            sample_rate,
            channels,
            frame_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    con.execute('BEGIN TRANSACTION')
    with tqdm(desc='Inserted noises', unit='rows') as pbar:
        for batch in batched(iter_noise_rows(), batch_size):
            con.executemany(insert_sql, batch)
            pbar.update(len(batch))
    con.execute('COMMIT')


def main() -> None:
    if not SQL_ROOT.exists():
        print_error(f'SQL database does not exist at {SQL_ROOT}')
        return
    else:
        print_log(f'Inserting noises from DEMAND into {SQL_ROOT}')

    con = db.connect(SQL_ROOT)

    insert_noises(con, batch_size=1_000)

    con.close()


if __name__ == '__main__':
    main()
