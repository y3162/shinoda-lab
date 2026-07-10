import argparse
from pathlib import Path

import duckdb as db
from tqdm import tqdm

from src.config import SQL_ROOT
from src.model.tokenizer import TokenizerFactory
from src.utils.print import print_error, print_log, print_warning

CORPUS_FILENAME = 'corpus_train.txt'

CORPUS_QUERY = """
SELECT ar.transcript
FROM asr_results AS ar
JOIN utterances AS u ON ar.utterance_id = u.id
WHERE u.dataset_name = 'librispeech'
    AND u.split LIKE 'train%'
    AND ar.model_name = ?
ORDER BY ar.id
"""

COUNT_QUERY = """
SELECT COUNT(*)
FROM asr_results AS ar
JOIN utterances AS u ON ar.utterance_id = u.id
WHERE u.dataset_name = 'librispeech'
    AND u.split LIKE 'train%'
    AND ar.model_name = ?
"""


def count_corpus_rows(
    con: db.DuckDBPyConnection,
    model_name: str,
) -> int:
    row = con.execute(COUNT_QUERY, [model_name]).fetchone()
    if row is None:
        return 0
    return int(row[0])


def export_corpus(
    con: db.DuckDBPyConnection,
    model_name: str,
    corpus_path: Path,
    *,
    total_rows: int | None = None,
    batch_size: int = 100_000,
) -> int:
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    result = con.execute(CORPUS_QUERY, [model_name])
    written = 0

    with corpus_path.open('w', encoding='utf-8') as f:
        progress = tqdm(total=total_rows, desc='Exporting corpus', unit='lines')
        while True:
            rows = result.fetchmany(batch_size)
            if not rows:
                break
            for (transcript,) in rows:
                line = transcript.replace('\n', ' ').replace('\r', ' ')
                f.write(line)
                f.write('\n')
                written += 1
            progress.update(len(rows))
        progress.close()

    return written


DEFAULT_INPUT_SENTENCE_SIZE = 5_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a tokenizer from asr_results transcripts (LibriSpeech train*).',
    )
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument(
        '--tokenizer_type',
        type=str,
        required=True,
        choices=['char', 'sentencepiece_bpe', 'sentencepiece_unigram'],
    )
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--vocab_size', type=int, default=None)
    parser.add_argument('--min_freq', type=int, default=1)
    parser.add_argument('--skip_unk_on_decode', action='store_true')
    parser.add_argument(
        '--corpus_path',
        type=Path,
        default=None,
        help='Use an existing corpus file instead of exporting from the database.',
    )
    parser.add_argument(
        '--force_export',
        action='store_true',
        help='Re-export the corpus even if corpus_path already exists.',
    )
    parser.add_argument(
        '--normalization_rule_name',
        type=str,
        default='identity',
    )
    parser.add_argument(
        '--character_coverage',
        type=float,
        default=1.0,
    )
    parser.add_argument(
        '--input_sentence_size',
        type=int,
        default=None,
        help=(
            'Number of sentences to sample for SentencePiece training. '
            f'Default: {DEFAULT_INPUT_SENTENCE_SIZE}. Set 0 to use the full corpus.'
        ),
    )
    parser.add_argument(
        '--shuffle_input_sentence',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Shuffle the corpus before sampling (used with --input_sentence_size).',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.tokenizer_type in {'sentencepiece_bpe', 'sentencepiece_unigram'}:
        if args.vocab_size is None:
            print_error('--vocab_size is required for SentencePiece tokenizers.')
            return
        if args.input_sentence_size is None:
            args.input_sentence_size = DEFAULT_INPUT_SENTENCE_SIZE
            print_log(
                f'Using default input_sentence_size={DEFAULT_INPUT_SENTENCE_SIZE} '
                'for SentencePiece training. Set --input_sentence_size 0 to use the full corpus.'
            )

    if not SQL_ROOT.exists():
        print_error(f'SQL database does not exist at {SQL_ROOT}')
        return

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = args.corpus_path or (output_dir / CORPUS_FILENAME)

    con = db.connect(SQL_ROOT, read_only=True)

    if args.corpus_path is None:
        row_count = count_corpus_rows(con, args.model_name)
        if row_count == 0:
            print_error(
                f'No asr_results found for model_name={args.model_name} '
                'in LibriSpeech train* splits.'
            )
            con.close()
            return

        if corpus_path.exists() and not args.force_export:
            print_warning(
                f'Corpus already exists at {corpus_path}. '
                'Skip export. Use --force_export to overwrite.'
            )
        else:
            print_log(
                f'Exporting {row_count} transcripts for model {args.model_name} '
                f'to {corpus_path}'
            )
            written = export_corpus(
                con,
                args.model_name,
                corpus_path,
                total_rows=row_count,
            )
            print_log(f'Exported {written} lines to {corpus_path}')
    else:
        if not corpus_path.exists():
            print_error(f'Corpus file not found: {corpus_path}')
            con.close()
            return
        print_log(f'Using existing corpus: {corpus_path}')

    con.close()

    train_kwargs: dict = {
        'skip_unk_on_decode': args.skip_unk_on_decode,
    }
    if args.tokenizer_type == 'char':
        train_kwargs['min_freq'] = args.min_freq
    else:
        train_kwargs['vocab_size'] = args.vocab_size
        train_kwargs['normalization_rule_name'] = args.normalization_rule_name
        train_kwargs['character_coverage'] = args.character_coverage
        if args.input_sentence_size > 0:
            train_kwargs['input_sentence_size'] = args.input_sentence_size
            train_kwargs['shuffle_input_sentence'] = args.shuffle_input_sentence

    print_log(
        f'Training {args.tokenizer_type} tokenizer into {output_dir}'
    )
    if args.tokenizer_type == 'char':
        print_log(
            'Building character vocabulary from corpus '
            '(large corpora may take around 15 minutes).'
        )
    tokenizer = TokenizerFactory.train(
        tokenizer_type=args.tokenizer_type,
        input_path=corpus_path,
        output_dir=output_dir,
        **train_kwargs,
    )
    print_log(
        f'Trained tokenizer: vocab_size={tokenizer.vocab_size}, '
        f'output_dir={output_dir}'
    )


if __name__ == '__main__':
    main()
