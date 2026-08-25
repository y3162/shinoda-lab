"""Insert additive noise_configs: clean, single-file, and dual-file mixes."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict

import duckdb as db
from tqdm import tqdm

from src.utils.print import (
    print_warning,
    print_log,
    print_error,
)
from src.config import SQL_ROOT


SNR_MIN = -10
SNR_MAX = 5

SPLIT_ORDER = (
    'train',
    'dev',
    'test',
)

SPLIT_RANK = {
    split: index
    for index, split in enumerate(SPLIT_ORDER)
}

DUAL_COUNTS = {
    'train': 200,
    'dev': 40,
    'test': 40,
}


def dumps_config(config: dict) -> str:
    """Convert a config to a canonical JSON string."""
    return json.dumps(
        config,
        sort_keys=True,
        separators=(',', ':'),
    )


def get_split_seed(
    seed: int,
    split: str,
) -> int:
    """Generate a process-independent seed for each split."""
    value = f'{seed}:{split}'.encode('utf-8')
    digest = hashlib.sha256(value).digest()

    return int.from_bytes(
        digest[:8],
        byteorder='big',
        signed=False,
    )


def insert_noise_config(
    con: db.DuckDBPyConnection,
    *,
    seed: int = 0,
) -> None:
    insert_sql = """
        INSERT OR IGNORE INTO noise_configs (
            config_json
        )
        VALUES (?)
    """

    rows = con.execute(
        """
        SELECT
            id,
            noise_type,
            split
        FROM noises
        """
    ).fetchall()

    if not rows:
        print_error('noises table is empty; insert clipped DEMAND first')
        raise SystemExit(1)

    rows = sorted(
        rows,
        key=lambda row: (
            SPLIT_RANK.get(str(row[2]), len(SPLIT_RANK)),
            str(row[2]),
            str(row[1]),
            int(row[0]),
        ),
    )

    by_split_type: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for noise_id, noise_type, split in rows:
        by_split_type[str(split)][str(noise_type)].append(int(noise_id))

    for type_to_ids in by_split_type.values():
        for noise_ids in type_to_ids.values():
            noise_ids.sort()

    con.execute('BEGIN TRANSACTION')

    try:
        with tqdm(desc='Inserted noise_configs', unit='rows') as pbar:
            # 1. Clean
            clean_config = {
                'generator_type': 'additive',
                'args': [],
            }

            con.execute(
                insert_sql,
                [dumps_config(clean_config)],
            )
            pbar.update(1)

            # 2. Single
            #
            # rows:
            # split -> noise_type -> noise_id
            #
            # snr:
            # -10 -> -9 -> ... -> 5
            for noise_id, noise_type, split in rows:
                for snr in range(SNR_MIN, SNR_MAX + 1):
                    config = {
                        'generator_type': 'additive',
                        'split': str(split),
                        'kind': 'single',
                        'args': [
                            {
                                'type': 'audiofile',
                                'noise_id': int(noise_id),
                                'snr_db': int(snr),
                            }
                        ],
                    }

                    con.execute(
                        insert_sql,
                        [dumps_config(config)],
                    )
                    pbar.update(1)

            # 3. Dual
            for split in SPLIT_ORDER:
                n_dual = DUAL_COUNTS[split]

                type_to_ids = by_split_type.get(split, {})
                noise_types = sorted(type_to_ids)

                if len(noise_types) < 2:
                    print_warning(
                        f'skip dual configs for split={split}: need >=2 types'
                    )
                    continue

                rng = random.Random(
                    get_split_seed(seed, split)
                )

                seen: set[
                    tuple[
                        tuple[str, int, int],
                        tuple[str, int, int],
                    ]
                ] = set()

                pairs: list[
                    tuple[
                        tuple[str, int, int],
                        tuple[str, int, int],
                    ]
                ] = []

                attempts = 0
                max_attempts = n_dual * 50

                while len(pairs) < n_dual and attempts < max_attempts:
                    attempts += 1

                    type_1, type_2 = rng.sample(noise_types, 2)

                    noise_id_1 = rng.choice(type_to_ids[type_1])
                    noise_id_2 = rng.choice(type_to_ids[type_2])

                    snr_1 = rng.randint(SNR_MIN, SNR_MAX)
                    snr_2 = rng.randint(SNR_MIN, SNR_MAX)

                    pair = tuple(sorted([
                        (
                            type_1,
                            int(noise_id_1),
                            int(snr_1),
                        ),
                        (
                            type_2,
                            int(noise_id_2),
                            int(snr_2),
                        ),
                    ]))

                    if pair in seen:
                        continue

                    seen.add(pair)
                    pairs.append(pair)

                if len(pairs) < n_dual:
                    print_warning(
                        f'dual configs for split={split}: '
                        f'only made {len(pairs)}/{n_dual}'
                    )

                pairs.sort()

                for pair in pairs:
                    (
                        (_, noise_id_1, snr_1),
                        (_, noise_id_2, snr_2),
                    ) = pair

                    config = {
                        'generator_type': 'additive',
                        'split': split,
                        'kind': 'dual',
                        'args': [
                            {
                                'type': 'audiofile',
                                'noise_id': noise_id_1,
                                'snr_db': snr_1,
                            },
                            {
                                'type': 'audiofile',
                                'noise_id': noise_id_2,
                                'snr_db': snr_2,
                            },
                        ],
                    }

                    con.execute(
                        insert_sql,
                        [dumps_config(config)],
                    )
                    pbar.update(1)

        con.execute('COMMIT')

    except Exception:
        con.execute('ROLLBACK')
        raise

    print_log('noise_configs insert committed')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not SQL_ROOT.exists():
        print_error(f'SQL database does not exist at {SQL_ROOT}')
        raise SystemExit(1)

    print_log(f'Inserting noise configs into {SQL_ROOT}')
    con = db.connect(str(SQL_ROOT))
    try:
        insert_noise_config(con, seed=args.seed)
    finally:
        con.close()


if __name__ == '__main__':
    main()
