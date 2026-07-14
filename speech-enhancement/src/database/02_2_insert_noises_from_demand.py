from typing import Iterable, Iterator, List, Tuple
NoiseRow = Tuple[str, str, str, int, int, int]

import duckdb as db
from tqdm import tqdm
import torchaudio

from src.utils.print import (
    print_log,
    print_error,
)
from src.utils.demand import (
    find_all_audio_files,
    get_noise_type,
)
from src.config import SQL_ROOT


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
    audio_files = find_all_audio_files()

    for audio_file in tqdm(
        audio_files,
        desc='Parsing audio files',
        unit='file',
    ):
        audio, sample_rate = torchaudio.load(audio_file)
        assert audio.dim() == 2, 'Audio must have 2 dimensions'
        channels = audio.shape[0]
        frame_count = audio.shape[1]
        yield (
            'demand',
            str(audio_file),
            get_noise_type(audio_file),
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
            sample_rate,
            channels,
            frame_count
        )
        VALUES (?, ?, ?, ?, ?, ?)
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
