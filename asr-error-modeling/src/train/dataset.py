from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import duckdb as db
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

from src.config import PARQUET_ROOT, SQL_ROOT
from src.model.tokenizer import PAD_ID, BaseTokenizer
from src.utils.print import print_log

Direction = Literal['clean2noisy', 'noisy2clean']

PAIR_CACHE_ROOT = PARQUET_ROOT / 'asr_pair_cache'

BASE_QUERY = """
WITH target_utterances AS (
    SELECT id
    FROM utterances
    WHERE utterances.split IN ({subset_placeholders})
),
clean_asr AS (
    SELECT
        asr_results.utterance_id,
        asr_results.transcript,
        ROW_NUMBER() OVER (
            PARTITION BY asr_results.utterance_id
            ORDER BY asr_results.noise_config_id
        ) AS rn
    FROM asr_results
    INNER JOIN target_utterances
        ON asr_results.utterance_id = target_utterances.id
    WHERE asr_results.noise_config_id IN (
        SELECT id
        FROM noise_configs
        WHERE json_array_length(config_json->'$.args') = 0
    )
)
SELECT
    noisy.utterance_id,
    noisy.noise_config_id,
    COALESCE(noises.noise_type, 'CLEAN') AS noise_type,
    COALESCE(
        CAST(json_extract_string(noise_configs.config_json, '$.args[0].snr_db') AS INTEGER),
        0
    ) AS snr_db,
    clean_asr.transcript AS clean_transcript,
    noisy.transcript AS noisy_transcript
FROM asr_results AS noisy
INNER JOIN target_utterances
    ON noisy.utterance_id = target_utterances.id
INNER JOIN clean_asr
    ON noisy.utterance_id = clean_asr.utterance_id
    AND clean_asr.rn = 1
INNER JOIN noise_configs
    ON noisy.noise_config_id = noise_configs.id
LEFT JOIN noises
    ON noises.id = CAST(
        json_extract_string(noise_configs.config_json, '$.args[0].noise_id') AS INTEGER
    )
WHERE noisy.noise_config_id IN ({config_placeholders})
    AND clean_asr.transcript IS NOT NULL
    AND clean_asr.transcript <> ''
    AND noisy.transcript IS NOT NULL
    AND noisy.transcript <> ''
ORDER BY noisy.utterance_id, noisy.noise_config_id
"""

CONDITION_QUERY = """
WITH target_utterances AS (
    SELECT id
    FROM utterances
    WHERE utterances.split IN ({subset_placeholders})
)
SELECT DISTINCT
    COALESCE(noises.noise_type, 'CLEAN') AS noise_type,
    COALESCE(
        CAST(json_extract_string(noise_configs.config_json, '$.args[0].snr_db') AS INTEGER),
        0
    ) AS snr_db
FROM asr_results AS noisy
INNER JOIN target_utterances
    ON noisy.utterance_id = target_utterances.id
INNER JOIN noise_configs
    ON noisy.noise_config_id = noise_configs.id
LEFT JOIN noises
    ON noises.id = CAST(
        json_extract_string(noise_configs.config_json, '$.args[0].noise_id') AS INTEGER
    )
WHERE noisy.noise_config_id IN ({config_placeholders})
    AND noisy.transcript IS NOT NULL
    AND noisy.transcript <> ''
ORDER BY noise_type, snr_db
"""

NON_CLEAN_NOISE_CONFIG_IDS_QUERY = """
SELECT id
FROM noise_configs
WHERE json_array_length(config_json->'$.args') <> 0
ORDER BY id
"""


@dataclass(frozen=True)
class AsrPairTextSample:
    utterance_id: int
    noise_config_id: int
    noise_type: str
    snr_db: int
    clean_transcript: str
    noisy_transcript: str


@dataclass(frozen=True)
class AsrPairTensorSample:
    utterance_id: int
    noise_config_id: int
    noise_type: str
    snr_db: int
    src_ids: torch.Tensor
    tgt_ids: torch.Tensor
    global_prefix_ids: torch.Tensor


@dataclass(frozen=True)
class PairCacheMeta:
    subsets: list[str]
    noise_config_ids: list[int]
    num_rows: int
    noise_types: list[str]
    snr_dbs: list[int]
    parquet_path: Path
    meta_path: Path


class ConditionVocab:
    def __init__(
        self,
        noise_types: list[str],
        snr_dbs: list[int],
        base_id: int,
    ) -> None:
        self.noise_types = list(noise_types)
        self.snr_dbs = list(snr_dbs)
        self.base_id = base_id
        self.noise_type_to_id = {
            noise_type: base_id + index
            for index, noise_type in enumerate(self.noise_types)
        }
        snr_base_id = base_id + len(self.noise_types)
        self.snr_db_to_id = {
            snr_db: snr_base_id + index
            for index, snr_db in enumerate(self.snr_dbs)
        }

    @property
    def num_tokens(self) -> int:
        return len(self.noise_types) + len(self.snr_dbs)

    def encode(self, noise_type: str, snr_db: int) -> list[int]:
        return [self.noise_type_to_id[noise_type], self.snr_db_to_id[snr_db]]

    def state_dict(self) -> dict:
        return {
            'noise_types': self.noise_types,
            'snr_dbs': self.snr_dbs,
            'base_id': self.base_id,
        }

    @classmethod
    def from_datasets(
        cls,
        datasets: list[AsrPairDataset],
        base_id: int,
    ) -> ConditionVocab:
        noise_types: set[str] = set()
        snr_dbs: set[int] = set()
        for dataset in datasets:
            noise_types.update(dataset.noise_types)
            snr_dbs.update(dataset.snr_dbs)
        return cls(
            noise_types=sorted(noise_types),
            snr_dbs=sorted(snr_dbs),
            base_id=base_id,
        )


def resolve_noise_config_ids(noise_config_ids: list[int] | None) -> list[int]:
    if noise_config_ids:
        return list(noise_config_ids)

    con = db.connect(SQL_ROOT, read_only=True)
    rows = con.execute(NON_CLEAN_NOISE_CONFIG_IDS_QUERY).fetchall()
    con.close()
    return [int(row[0]) for row in rows]


def _format_sql(query: str, subsets: list[str], noise_config_ids: list[int]) -> tuple[str, list]:
    sql = query.format(
        subset_placeholders=', '.join(['?' for _ in subsets]),
        config_placeholders=', '.join(['?' for _ in noise_config_ids]),
    )
    return sql, [*subsets, *noise_config_ids]


def _cache_key(subsets: list[str], noise_config_ids: list[int]) -> str:
    payload = {
        'subsets': sorted(subsets),
        'noise_config_ids': sorted(noise_config_ids),
    }
    digest = hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8'),
    ).hexdigest()
    return digest


def _escape_sql_string(path: Path) -> str:
    return str(path).replace("'", "''")


def ensure_asr_pair_cache(
    subsets: list[str],
    noise_config_ids: list[int],
) -> PairCacheMeta:
    cache_dir = PAIR_CACHE_ROOT / _cache_key(subsets, noise_config_ids)
    parquet_path = cache_dir / 'pairs.parquet'
    meta_path = cache_dir / 'meta.json'

    if parquet_path.exists() and meta_path.exists():
        with meta_path.open(encoding='utf-8') as f:
            meta = json.load(f)
        num_rows = int(meta['num_rows'])
        print_log(
            f'Using ASR pair cache: path={parquet_path}, '
            f'rows={num_rows}, subsets={list(meta["subsets"])}, '
            f'noise_config_ids={len(meta["noise_config_ids"])}'
        )
        return PairCacheMeta(
            subsets=list(meta['subsets']),
            noise_config_ids=[int(x) for x in meta['noise_config_ids']],
            num_rows=num_rows,
            noise_types=[str(x) for x in meta['noise_types']],
            snr_dbs=[int(x) for x in meta['snr_dbs']],
            parquet_path=parquet_path,
            meta_path=meta_path,
        )

    print_log(
        f'Creating ASR pair cache: path={parquet_path}, '
        f'subsets={list(subsets)}, noise_config_ids={len(noise_config_ids)}'
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_parquet_path = cache_dir / 'pairs.parquet.tmp'
    if tmp_parquet_path.exists():
        tmp_parquet_path.unlink()

    pair_sql, pair_params = _format_sql(BASE_QUERY, subsets, noise_config_ids)
    condition_sql, condition_params = _format_sql(
        CONDITION_QUERY,
        subsets,
        noise_config_ids,
    )

    con = db.connect(SQL_ROOT, read_only=True)
    con.execute(
        f"COPY ({pair_sql}) TO '{_escape_sql_string(tmp_parquet_path)}' (FORMAT PARQUET)",
        pair_params,
    )
    condition_rows = con.execute(condition_sql, condition_params).fetchall()
    con.close()

    tmp_parquet_path.replace(parquet_path)
    num_rows = int(pq.ParquetFile(parquet_path).metadata.num_rows)
    noise_types = sorted({str(row[0]) for row in condition_rows})
    snr_dbs = sorted({int(row[1]) for row in condition_rows})

    meta = {
        'subsets': list(subsets),
        'noise_config_ids': list(noise_config_ids),
        'num_rows': num_rows,
        'noise_types': noise_types,
        'snr_dbs': snr_dbs,
    }
    with meta_path.open('w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write('\n')

    print_log(
        f'Created ASR pair cache: path={parquet_path}, rows={num_rows}'
    )
    return PairCacheMeta(
        subsets=list(subsets),
        noise_config_ids=list(noise_config_ids),
        num_rows=num_rows,
        noise_types=noise_types,
        snr_dbs=snr_dbs,
        parquet_path=parquet_path,
        meta_path=meta_path,
    )


def _build_row_group_offsets(parquet_path: Path) -> np.ndarray:
    metadata = pq.read_metadata(parquet_path)
    offsets = np.empty(metadata.num_row_groups + 1, dtype=np.int64)
    offsets[0] = 0
    for row_group_index in range(metadata.num_row_groups):
        offsets[row_group_index + 1] = (
            offsets[row_group_index]
            + metadata.row_group(row_group_index).num_rows
        )
    return offsets


class AsrPairDataset(Dataset):
    def __init__(
        self,
        subsets: list[str],
        noise_config_ids: list[int] | None,
    ) -> None:
        resolved_ids = resolve_noise_config_ids(noise_config_ids)
        self.noise_config_ids = resolved_ids
        self._cache = ensure_asr_pair_cache(subsets, resolved_ids)
        self.noise_types = list(self._cache.noise_types)
        self.snr_dbs = list(self._cache.snr_dbs)
        self.parquet_path = self._cache.parquet_path
        self.row_group_offsets = _build_row_group_offsets(self.parquet_path)
        self._local = threading.local()

    def __len__(self) -> int:
        return self._cache.num_rows

    def _parquet_file(self) -> pq.ParquetFile:
        parquet_file = getattr(self._local, 'parquet_file', None)
        if parquet_file is None:
            parquet_file = pq.ParquetFile(self.parquet_path, memory_map=True)
            self._local.parquet_file = parquet_file
        return parquet_file

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= self._cache.num_rows:
            raise IndexError(f'index {index} out of range for size {self._cache.num_rows}')
        row_group_index = int(
            np.searchsorted(self.row_group_offsets[1:], index, side='right')
        )
        local_index = index - int(self.row_group_offsets[row_group_index])
        return row_group_index, local_index

    def _row_group_table(self, row_group_index: int) -> pa.Table:
        cache = getattr(self._local, 'row_group_cache', None)
        if cache is not None and cache[0] == row_group_index:
            return cache[1]

        table = self._parquet_file().read_row_group(row_group_index)
        self._local.row_group_cache = (row_group_index, table)
        return table

    def __getitem__(self, index: int) -> AsrPairTextSample:
        row_group_index, local_index = self._locate(index)
        table = self._row_group_table(row_group_index)
        return AsrPairTextSample(
            utterance_id=int(table.column('utterance_id')[local_index].as_py()),
            noise_config_id=int(table.column('noise_config_id')[local_index].as_py()),
            noise_type=str(table.column('noise_type')[local_index].as_py()),
            snr_db=int(table.column('snr_db')[local_index].as_py()),
            clean_transcript=str(table.column('clean_transcript')[local_index].as_py()),
            noisy_transcript=str(table.column('noisy_transcript')[local_index].as_py()),
        )


class RowGroupShuffleSampler(Sampler[int]):
    def __init__(
        self,
        row_group_offsets: np.ndarray,
        generator: torch.Generator | None = None,
    ) -> None:
        if row_group_offsets.ndim != 1 or row_group_offsets.size < 2:
            raise ValueError('row_group_offsets must be a 1D array with at least 2 entries')
        self.row_group_offsets = np.asarray(row_group_offsets, dtype=np.int64)
        self.generator = generator
        self.num_samples = int(self.row_group_offsets[-1])

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        num_row_groups = self.row_group_offsets.size - 1
        row_group_order = torch.randperm(
            num_row_groups,
            generator=self.generator,
        ).tolist()
        for row_group_index in row_group_order:
            start = int(self.row_group_offsets[row_group_index])
            end = int(self.row_group_offsets[row_group_index + 1])
            local_count = end - start
            local_order = torch.randperm(
                local_count,
                generator=self.generator,
            ).tolist()
            for local_index in local_order:
                yield start + local_index


def truncate_token_ids(token_ids: list[int], max_length: int) -> list[int]:
    if max_length <= 0:
        raise ValueError(f'max_length must be positive, got {max_length}')
    if len(token_ids) <= max_length:
        return token_ids
    return token_ids[:max_length]


class AsrPairTokenizingDataset(Dataset):
    def __init__(
        self,
        dataset: AsrPairDataset,
        tokenizer: BaseTokenizer,
        condition_vocab: ConditionVocab,
        direction: Direction,
        context_length: int,
        global_prefix_len: int = 2,
    ) -> None:
        if context_length <= global_prefix_len:
            raise ValueError(
                'context_length must be greater than global_prefix_len '
                f'({global_prefix_len}), got {context_length}'
            )
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.condition_vocab = condition_vocab
        self.direction = direction
        self.context_length = context_length
        self.global_prefix_len = global_prefix_len
        self.max_src_tokens = context_length - global_prefix_len
        self.max_tgt_tokens = context_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> AsrPairTensorSample:
        sample = self.dataset[index]
        if self.direction == 'clean2noisy':
            src_text = sample.clean_transcript
            tgt_text = sample.noisy_transcript
        else:
            src_text = sample.noisy_transcript
            tgt_text = sample.clean_transcript

        src_ids = truncate_token_ids(
            self.tokenizer.encode(src_text),
            self.max_src_tokens,
        )
        tgt_ids = truncate_token_ids(
            self.tokenizer.encode(tgt_text),
            self.max_tgt_tokens,
        )
        global_prefix_ids = torch.tensor(
            self.condition_vocab.encode(sample.noise_type, sample.snr_db),
            dtype=torch.long,
        )
        if global_prefix_ids.numel() != self.global_prefix_len:
            raise ValueError(
                'global_prefix_len mismatch: '
                f'expected {self.global_prefix_len}, got {global_prefix_ids.numel()}'
            )
        return AsrPairTensorSample(
            utterance_id=sample.utterance_id,
            noise_config_id=sample.noise_config_id,
            noise_type=sample.noise_type,
            snr_db=sample.snr_db,
            src_ids=torch.tensor(src_ids, dtype=torch.long),
            tgt_ids=torch.tensor(tgt_ids, dtype=torch.long),
            global_prefix_ids=global_prefix_ids,
        )


def collate_samples(
    batch: list[AsrPairTensorSample],
) -> dict[str, torch.Tensor | list[int] | list[str]]:
    max_src_len = max(sample.src_ids.size(0) for sample in batch)
    max_tgt_len = max(sample.tgt_ids.size(0) for sample in batch)
    src_ids = torch.full((len(batch), max_src_len), PAD_ID, dtype=torch.long)
    tgt_ids = torch.full((len(batch), max_tgt_len), PAD_ID, dtype=torch.long)
    global_prefix_ids = torch.stack(
        [sample.global_prefix_ids for sample in batch],
        dim=0,
    )
    for index, sample in enumerate(batch):
        src_ids[index, :sample.src_ids.size(0)] = sample.src_ids
        tgt_ids[index, :sample.tgt_ids.size(0)] = sample.tgt_ids
    return {
        'utterance_ids': [sample.utterance_id for sample in batch],
        'noise_config_ids': [sample.noise_config_id for sample in batch],
        'noise_types': [sample.noise_type for sample in batch],
        'snr_dbs': [sample.snr_db for sample in batch],
        'src_ids': src_ids,
        'tgt_ids': tgt_ids,
        'global_prefix_ids': global_prefix_ids,
    }
