from typing import Iterable, Iterator, List, Tuple
UtteranceRow = Tuple[str, str, str, str, str, str, int, int, int]
from tqdm import tqdm

import duckdb as db
import torchaudio

from src.utils.print import (
    print_log,
    print_error,
)
from src.utils.librispeech import (
    find_all_transcript_files,
    parse_transcript_file,
    get_subset_name,
    get_speaker_id,
    get_section_id,
)
from src.config import PROJECT_ROOT, SQL_ROOT, resolve_project_path


def batched(
    iterable: Iterable[UtteranceRow],
    batch_size: int,
) -> Iterator[List[UtteranceRow]]:
    batch = []

    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def iter_utterance_rows() -> Iterator[UtteranceRow]:
    transcript_files = sorted(
        find_all_transcript_files(),
        key=lambda path: str(path),
    )

    for transcript_file in tqdm(
        transcript_files,
        desc='Parsing transcript files',
        unit='file',
    ):
        utterances = sorted(
            parse_transcript_file(transcript_file),
            key=lambda item: str(item[0]),  # audio_path
        )

        for audio_path, transcript in utterances:
            audio_path = resolve_project_path(audio_path)
            audio, sample_rate = torchaudio.load(audio_path)

            assert audio.dim() == 2, 'Audio must have 2 dimensions'

            channels = audio.shape[0]
            frame_count = audio.shape[1]

            yield (
                'librispeech',
                get_subset_name(audio_path),
                str(audio_path.relative_to(PROJECT_ROOT)),
                get_speaker_id(audio_path),
                get_section_id(audio_path),
                transcript,
                sample_rate,
                channels,
                frame_count,
            )


def insert_utterances(
    con: db.DuckDBPyConnection,
    batch_size: int = 1_000,
) -> None:
    insert_sql = """
        INSERT INTO utterances (
            dataset_name,
            split,
            audio_path,
            speaker_id,
            section_id,
            transcript,
            sample_rate,
            channels,
            frame_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    con.execute('BEGIN TRANSACTION')
    with tqdm(desc='Inserted utterances', unit='rows') as pbar:
        for batch in batched(iter_utterance_rows(), batch_size):
            con.executemany(insert_sql, batch)
            pbar.update(len(batch))
    con.execute('COMMIT')


def main() -> None:
    if not SQL_ROOT.exists():
        print_error(f'SQL database does not exist at {SQL_ROOT}')
        return
    else:
        print_log(f'Inserting utterances from LibriSpeech into {SQL_ROOT}')

    con = db.connect(SQL_ROOT)

    insert_utterances(con, batch_size=1_000)

    con.close()


if __name__ == '__main__':
    main()
