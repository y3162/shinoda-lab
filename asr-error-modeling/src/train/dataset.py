from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import duckdb as db
import torch
from torch.utils.data import Dataset

from src.config import SQL_ROOT
from src.model.tokenizer import PAD_ID, BaseTokenizer

Direction = Literal['clean2noisy', 'noisy2clean']

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
            for row in dataset.rows:
                noise_types.add(str(row[2]))
                snr_dbs.add(int(row[3]))
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


def fetch_asr_pair_rows(
    subsets: list[str],
    noise_config_ids: list[int],
) -> list[tuple]:
    con = db.connect(SQL_ROOT, read_only=True)

    sql_params = {
        'subset_placeholders': ', '.join(['?' for _ in subsets]),
        'config_placeholders': ', '.join(['?' for _ in noise_config_ids]),
    }
    rows = con.execute(
        BASE_QUERY.format(**sql_params),
        [*subsets, *noise_config_ids],
    ).fetchall()

    con.close()
    return rows


class AsrPairDataset(Dataset):
    def __init__(
        self,
        subsets: list[str],
        noise_config_ids: list[int] | None,
    ) -> None:
        resolved_ids = resolve_noise_config_ids(noise_config_ids)
        self.noise_config_ids = resolved_ids
        self.rows = fetch_asr_pair_rows(subsets, resolved_ids)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> AsrPairTextSample:
        (
            utterance_id,
            noise_config_id,
            noise_type,
            snr_db,
            clean_transcript,
            noisy_transcript,
        ) = self.rows[index]
        return AsrPairTextSample(
            utterance_id=utterance_id,
            noise_config_id=noise_config_id,
            noise_type=noise_type,
            snr_db=snr_db,
            clean_transcript=clean_transcript,
            noisy_transcript=noisy_transcript,
        )


class AsrPairTokenizingDataset(Dataset):
    def __init__(
        self,
        dataset: AsrPairDataset,
        tokenizer: BaseTokenizer,
        condition_vocab: ConditionVocab,
        direction: Direction,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.condition_vocab = condition_vocab
        self.direction = direction

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

        src_ids = torch.tensor(
            self.tokenizer.encode(src_text),
            dtype=torch.long,
        )
        tgt_ids = torch.tensor(
            self.tokenizer.encode(tgt_text),
            dtype=torch.long,
        )
        global_prefix_ids = torch.tensor(
            self.condition_vocab.encode(sample.noise_type, sample.snr_db),
            dtype=torch.long,
        )
        return AsrPairTensorSample(
            utterance_id=sample.utterance_id,
            noise_config_id=sample.noise_config_id,
            noise_type=sample.noise_type,
            snr_db=sample.snr_db,
            src_ids=src_ids,
            tgt_ids=tgt_ids,
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
