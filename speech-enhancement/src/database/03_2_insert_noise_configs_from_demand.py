"""Insert additive noise_configs: clean, single-file, and dual-file mixes."""
from __future__ import annotations

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
DUAL_COUNTS = {
    'train': 200,
    'dev': 40,
    'test': 40,
}


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
        SELECT id, noise_type, split
        FROM noises
        ORDER BY noise_type, split, id
        """
    ).fetchall()
    if not rows:
        print_error('noises table is empty; insert clipped DEMAND first')
        raise SystemExit(1)

    by_split_type: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for noise_id, noise_type, split in rows:
        by_split_type[split][noise_type].append(int(noise_id))

    con.execute('BEGIN TRANSACTION')
    with tqdm(desc='Inserted noise_configs', unit='rows') as pbar:
        # Clean (no noise)
        con.execute(insert_sql, [json.dumps(
            {
                'generator_type': 'additive',
                'args': [],
            }
        )])
        pbar.update(1)

        # Single-file additive: every clip x SNR
        for noise_id, noise_type, split in rows:
            for snr in range(SNR_MIN, SNR_MAX + 1):
                con.execute(insert_sql, [json.dumps(
                    {
                        'generator_type': 'additive',
                        'split': split,
                        'kind': 'single',
                        'args': [
                            {
                                'type': 'audiofile',
                                'noise_id': int(noise_id),
                                'snr_db': int(snr),
                            }
                        ],
                    }
                )])
                pbar.update(1)

        # Dual-file additive: two different noise_types, same split
        for split, n_dual in DUAL_COUNTS.items():
            type_to_ids = by_split_type.get(split, {})
            types = sorted(type_to_ids.keys())
            if len(types) < 2:
                print_warning(f'skip dual configs for split={split}: need >=2 types')
                continue
            rng = random.Random(seed + hash(split) % 10_000_000)
            seen: set[tuple] = set()
            made = 0
            attempts = 0
            max_attempts = n_dual * 50
            while made < n_dual and attempts < max_attempts:
                attempts += 1
                t1, t2 = rng.sample(types, 2)
                id1 = rng.choice(type_to_ids[t1])
                id2 = rng.choice(type_to_ids[t2])
                snr1 = rng.randint(SNR_MIN, SNR_MAX)
                snr2 = rng.randint(SNR_MIN, SNR_MAX)
                # Canonicalize order for dedup
                pair = tuple(sorted([
                    (t1, id1, snr1),
                    (t2, id2, snr2),
                ]))
                if pair in seen:
                    continue
                seen.add(pair)
                (ta, ida, snra), (tb, idb, snrb) = pair
                con.execute(insert_sql, [json.dumps(
                    {
                        'generator_type': 'additive',
                        'split': split,
                        'kind': 'dual',
                        'args': [
                            {
                                'type': 'audiofile',
                                'noise_id': int(ida),
                                'snr_db': int(snra),
                            },
                            {
                                'type': 'audiofile',
                                'noise_id': int(idb),
                                'snr_db': int(snrb),
                            },
                        ],
                    }
                )])
                made += 1
                pbar.update(1)
            if made < n_dual:
                print_warning(
                    f'dual configs for split={split}: only made {made}/{n_dual}'
                )

    con.execute('COMMIT')
    print_log('noise_configs insert committed')


def main() -> None:
    if SQL_ROOT.exists():
        print_warning(f'SQL database already exists at {SQL_ROOT}')
    else:
        print_log(f'Creating SQL database at {SQL_ROOT}')
        SQL_ROOT.parent.mkdir(parents=True, exist_ok=True)

    con = db.connect(SQL_ROOT)

    insert_noise_config(con)

    con.close()


if __name__ == '__main__':
    main()
