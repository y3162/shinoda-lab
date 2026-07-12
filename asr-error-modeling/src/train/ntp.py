from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.model.tokenizer import BaseTokenizer, TokenizerFactory
from src.model.transformer import Seq2SeqTransformer
from src.train.dataset import (
    AsrPairDataset,
    AsrPairTokenizingDataset,
    ConditionVocab,
    Direction,
    RowGroupShuffleSampler,
    collate_samples,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / 'configs' / 'clean2noisy_base.json'
VALID_DIRECTIONS = ('clean2noisy', 'noisy2clean')


@dataclass(frozen=True)
class ModelConfig:
    n_encoder_layers: int
    n_decoder_layers: int
    d_model: int
    n_heads: int
    context_length: int


@dataclass(frozen=True)
class TrainConfig:
    train_subsets: list[str]
    valid_subsets: list[str]
    tokenizer_path: Path
    noise_config_ids: list[int] | None
    batch_size: int
    num_workers: int
    learning_rate: float
    epochs: int
    seed: int
    summary_interval: int


@dataclass(frozen=True)
class EnvConfig:
    output_dir: Path


@dataclass(frozen=True)
class NtpConfig:
    config_path: Path
    direction: Direction
    model: ModelConfig
    train: TrainConfig
    env: EnvConfig


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding='utf-8') as f:
        return json.load(f)


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_direction(value: str) -> Direction:
    if value not in VALID_DIRECTIONS:
        raise ValueError(
            f'Invalid direction: {value!r}. Expected one of {VALID_DIRECTIONS}.'
        )
    return value  # type: ignore[return-value]


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def parse_noise_config_ids(value: list[int] | None) -> list[int] | None:
    if value is None or len(value) == 0:
        return None
    return list(value)


def parse_args(default_config_path: Path = DEFAULT_CONFIG_PATH) -> NtpConfig:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=Path, default=default_config_path)
    pre_args, remaining = pre_parser.parse_known_args()

    config = load_config(pre_args.config)
    model_cfg = config['model']
    train_cfg = config['train']
    env_cfg = config['env']

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=Path,
        default=pre_args.config,
        help='Path to the training config JSON.',
    )
    parser.add_argument(
        '--direction',
        type=str,
        choices=list(VALID_DIRECTIONS),
        default=config['direction'],
    )
    parser.add_argument(
        '--n_encoder_layers',
        type=int,
        default=model_cfg['n_encoder_layers'],
    )
    parser.add_argument(
        '--n_decoder_layers',
        type=int,
        default=model_cfg['n_decoder_layers'],
    )
    parser.add_argument(
        '--d_model',
        type=int,
        default=model_cfg['d_model'],
    )
    parser.add_argument(
        '--n_heads',
        type=int,
        default=model_cfg['n_heads'],
    )
    parser.add_argument(
        '--context_length',
        type=int,
        default=model_cfg['context_length'],
    )
    parser.add_argument(
        '--train_subsets',
        type=str,
        nargs='+',
        default=train_cfg['train_subsets'],
    )
    parser.add_argument(
        '--valid_subsets',
        type=str,
        nargs='+',
        default=train_cfg['valid_subsets'],
    )
    parser.add_argument(
        '--tokenizer_path',
        type=Path,
        default=train_cfg['tokenizer_path'],
    )
    parser.add_argument(
        '--noise_config_ids',
        type=int,
        nargs='+',
        default=train_cfg.get('noise_config_ids'),
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=train_cfg['batch_size'],
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=train_cfg['num_workers'],
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=train_cfg['learning_rate'],
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=train_cfg['epochs'],
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=train_cfg['seed'],
    )
    parser.add_argument(
        '--summary_interval',
        type=int,
        default=train_cfg.get('summary_interval', 100),
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=env_cfg['output_dir'],
    )
    args = parser.parse_args(remaining)

    return NtpConfig(
        config_path=args.config,
        direction=parse_direction(args.direction),
        model=ModelConfig(
            n_encoder_layers=args.n_encoder_layers,
            n_decoder_layers=args.n_decoder_layers,
            d_model=args.d_model,
            n_heads=args.n_heads,
            context_length=args.context_length,
        ),
        train=TrainConfig(
            train_subsets=args.train_subsets,
            valid_subsets=args.valid_subsets,
            tokenizer_path=resolve_path(args.tokenizer_path),
            noise_config_ids=parse_noise_config_ids(args.noise_config_ids),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            seed=args.seed,
            summary_interval=args.summary_interval,
        ),
        env=EnvConfig(
            output_dir=resolve_path(args.output_dir),
        ),
    )


def setup_logging(log_dir: Path, direction: Direction) -> logging.Logger:
    logger = logging.getLogger(f'ntp.{direction}.train')
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    file_handler = logging.FileHandler(log_dir / 'train.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def format_postfix(**kwargs) -> dict:
    formatted = {}
    for key, value in kwargs.items():
        if isinstance(value, float):
            formatted[key] = f'{value:.3f}'
        else:
            formatted[key] = value
    return formatted


def resolve_checkpoint_path(checkpoint_root: Path) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return checkpoint_root / timestamp


def config_to_dict(
    config: NtpConfig,
    noise_config_ids: list[int],
) -> dict:
    return {
        'direction': config.direction,
        'model': {
            'n_encoder_layers': config.model.n_encoder_layers,
            'n_decoder_layers': config.model.n_decoder_layers,
            'd_model': config.model.d_model,
            'n_heads': config.model.n_heads,
            'context_length': config.model.context_length,
        },
        'train': {
            'train_subsets': config.train.train_subsets,
            'valid_subsets': config.train.valid_subsets,
            'tokenizer_path': str(config.train.tokenizer_path),
            'noise_config_ids': noise_config_ids,
            'batch_size': config.train.batch_size,
            'num_workers': config.train.num_workers,
            'learning_rate': config.train.learning_rate,
            'epochs': config.train.epochs,
            'seed': config.train.seed,
            'summary_interval': config.train.summary_interval,
        },
        'env': {
            'output_dir': str(config.env.output_dir),
        },
    }


def build_env(
    config: NtpConfig,
    checkpoint_path: Path,
    noise_config_ids: list[int],
) -> None:
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    (checkpoint_path / 'logs').mkdir(parents=True, exist_ok=True)
    with (checkpoint_path / 'config.json').open('w', encoding='utf-8') as f:
        json.dump(
            config_to_dict(config, noise_config_ids),
            f,
            ensure_ascii=False,
            indent=4,
        )
        f.write('\n')


def save_checkpoint(filepath: Path, obj: dict) -> None:
    torch.save(obj, filepath)


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
) -> float:
    labels = targets[:, 1:].contiguous()
    preds = logits.argmax(dim=-1)
    mask = labels.ne(pad_id)
    if mask.sum().item() == 0:
        return 0.0
    correct = preds.eq(labels) & mask
    return (correct.sum().float() / mask.sum().float()).item()


def compute_batch_metrics(
    loss: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
) -> dict[str, float]:
    loss_value = float(loss.detach())
    return {
        'loss': loss_value,
        'ppl': math.exp(min(loss_value, 100.0)),
        'acc': compute_accuracy(logits.detach(), targets, pad_id),
    }


def batch_context_lengths(
    batch: dict[str, torch.Tensor | list[int] | list[str]],
) -> tuple[int, int]:
    src_len = int(batch['src_ids'].size(1)) + int(batch['global_prefix_ids'].size(1))
    tgt_len = int(batch['tgt_ids'].size(1))
    return src_len, tgt_len


def build_datasets(
    config: NtpConfig,
) -> tuple[AsrPairDataset, AsrPairDataset]:
    train = config.train
    train_dataset = AsrPairDataset(train.train_subsets, train.noise_config_ids)
    valid_dataset = AsrPairDataset(train.valid_subsets, train.noise_config_ids)
    return train_dataset, valid_dataset


def build_dataloaders(
    config: NtpConfig,
    generator: torch.Generator,
    tokenizer: BaseTokenizer,
    condition_vocab: ConditionVocab,
    train_dataset: AsrPairDataset,
    valid_dataset: AsrPairDataset,
    direction: Direction,
) -> tuple[DataLoader, DataLoader]:
    train = config.train
    train_tokenizing_dataset = AsrPairTokenizingDataset(
        train_dataset,
        tokenizer,
        condition_vocab,
        direction=direction,
        context_length=config.model.context_length,
    )
    valid_tokenizing_dataset = AsrPairTokenizingDataset(
        valid_dataset,
        tokenizer,
        condition_vocab,
        direction=direction,
        context_length=config.model.context_length,
    )
    dataloader_kwargs = {
        'batch_size': train.batch_size,
        'num_workers': train.num_workers,
        'collate_fn': collate_samples,
        'worker_init_fn': seed_worker,
        'generator': generator,
    }
    train_dataloader = DataLoader(
        train_tokenizing_dataset,
        sampler=RowGroupShuffleSampler(
            train_dataset.row_group_offsets,
            generator=generator,
        ),
        drop_last=True,
        **dataloader_kwargs,
    )
    valid_dataloader = DataLoader(
        valid_tokenizing_dataset,
        shuffle=False,
        drop_last=False,
        **dataloader_kwargs,
    )
    return train_dataloader, valid_dataloader


def build_models(
    config: NtpConfig,
    tokenizer: BaseTokenizer,
    condition_vocab: ConditionVocab,
) -> tuple[Seq2SeqTransformer, torch.optim.Optimizer]:
    transformer = Seq2SeqTransformer(
        vocab_size=tokenizer.vocab_size + condition_vocab.num_tokens,
        n_encoder_layers=config.model.n_encoder_layers,
        n_decoder_layers=config.model.n_decoder_layers,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        context_length=config.model.context_length,
        pad_id=tokenizer.pad_id,
    )
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=config.train.learning_rate)
    return transformer, optimizer


def forward_batch(
    transformer: Seq2SeqTransformer,
    batch: dict[str, torch.Tensor | list[int] | list[str]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_ids = batch['src_ids'].to(device)
    tgt_ids = batch['tgt_ids'].to(device)
    global_prefix_ids = batch['global_prefix_ids'].to(device)
    src_ids = transformer.add_global_prefix(src_ids, global_prefix_ids)
    logits = transformer(
        src_ids,
        tgt_ids,
        global_prefix_len=global_prefix_ids.size(1),
    )
    return logits, tgt_ids


def run_validation(
    transformer: Seq2SeqTransformer,
    valid_dataloader: DataLoader,
    device: torch.device,
    pad_id: int,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    transformer.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    val_pbar = tqdm(
        total=len(valid_dataloader),
        unit='batch',
        desc='Validation {}/{}'.format(epoch + 1, total_epochs),
        dynamic_ncols=True,
        leave=False,
    )

    with torch.no_grad():
        for batch in valid_dataloader:
            src_len, tgt_len = batch_context_lengths(batch)
            logits, tgt_ids = forward_batch(transformer, batch, device)
            loss = transformer.cross_entropy_loss(logits, tgt_ids)
            metrics = compute_batch_metrics(loss, logits, tgt_ids, pad_id)
            total_loss += metrics['loss']
            total_acc += metrics['acc']
            num_batches += 1
            val_pbar.set_postfix(
                format_postfix(
                    loss=metrics['loss'],
                    ppl=metrics['ppl'],
                    acc=metrics['acc'],
                    ctx=f'{src_len}/{tgt_len}',
                ),
                refresh=False,
            )
            val_pbar.update(1)

    val_pbar.close()

    if num_batches == 0:
        return {'loss': 0.0, 'ppl': 1.0, 'acc': 0.0}

    mean_loss = total_loss / num_batches
    return {
        'loss': mean_loss,
        'ppl': math.exp(min(mean_loss, 100.0)),
        'acc': total_acc / num_batches,
    }


def run_training(default_config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    config = parse_args(default_config_path)
    direction = config.direction
    generator = set_seed(config.train.seed)

    checkpoint_path = resolve_checkpoint_path(config.env.output_dir)
    tokenizer = TokenizerFactory.load(config.train.tokenizer_path)
    train_dataset, valid_dataset = build_datasets(config)
    noise_config_ids = train_dataset.noise_config_ids
    build_env(config, checkpoint_path, noise_config_ids)
    logger = setup_logging(checkpoint_path / 'logs', direction)
    sw = SummaryWriter(str(checkpoint_path / 'logs'))

    condition_vocab = ConditionVocab.from_datasets(
        [train_dataset, valid_dataset],
        base_id=tokenizer.vocab_size,
    )
    train_dataloader, valid_dataloader = build_dataloaders(
        config,
        generator,
        tokenizer,
        condition_vocab,
        train_dataset,
        valid_dataset,
        direction=direction,
    )
    transformer, optimizer = build_models(config, tokenizer, condition_vocab)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformer.to(device)

    logger.info('direction: %s', direction)
    logger.info('checkpoints directory: %s', checkpoint_path)
    logger.info('device: %s', device)
    logger.info('train samples: %d', len(train_dataset))
    logger.info('valid samples: %d', len(valid_dataset))
    logger.info('train pair cache: %s', train_dataset.parquet_path)
    logger.info('valid pair cache: %s', valid_dataset.parquet_path)
    logger.info('noise_config_ids: %s', noise_config_ids)
    logger.info(
        'Total Parameters: %.3fM',
        sum(p.numel() for p in transformer.parameters()) / 1e6,
    )
    tqdm.write('direction: {}'.format(direction))
    tqdm.write('checkpoints directory: {}'.format(checkpoint_path))
    tqdm.write('noise_config_ids count: {}'.format(len(noise_config_ids)))

    best_val_loss = float('inf')
    steps = 0

    for epoch in range(config.train.epochs):
        start = time.time()
        logger.info('Epoch: %d', epoch + 1)
        transformer.train()

        train_pbar = tqdm(
            total=len(train_dataloader),
            unit='batch',
            desc='Epoch {}/{}'.format(epoch + 1, config.train.epochs),
            dynamic_ncols=True,
        )

        for batch in train_dataloader:
            src_len, tgt_len = batch_context_lengths(batch)
            logits, tgt_ids = forward_batch(transformer, batch, device)
            loss = transformer.cross_entropy_loss(logits, tgt_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = compute_batch_metrics(loss, logits, tgt_ids, tokenizer.pad_id)
            if steps % config.train.summary_interval == 0:
                sw.add_scalar('Training/Loss', metrics['loss'], steps)
                sw.add_scalar('Training/PPL', metrics['ppl'], steps)
                sw.add_scalar('Training/Accuracy', metrics['acc'], steps)

            train_pbar.set_postfix(
                format_postfix(
                    loss=metrics['loss'],
                    ppl=metrics['ppl'],
                    acc=metrics['acc'],
                    ctx=f'{src_len}/{tgt_len}',
                ),
                refresh=False,
            )
            train_pbar.update(1)
            steps += 1

        train_pbar.close()

        val_metrics = run_validation(
            transformer,
            valid_dataloader,
            device,
            tokenizer.pad_id,
            epoch=epoch,
            total_epochs=config.train.epochs,
        )

        val_message = (
            'Validation (epoch {}/{}): LOSS={:.3f}, PPL={:.3f}, ACC={:.3f}'
        ).format(
            epoch + 1,
            config.train.epochs,
            val_metrics['loss'],
            val_metrics['ppl'],
            val_metrics['acc'],
        )
        tqdm.write(val_message)
        logger.info(val_message)

        sw.add_scalar('Validation/Loss', val_metrics['loss'], epoch + 1)
        sw.add_scalar('Validation/PPL', val_metrics['ppl'], epoch + 1)
        sw.add_scalar('Validation/Accuracy', val_metrics['acc'], epoch + 1)

        checkpoint_payload = {
            'model': transformer.state_dict(),
            'optimizer': optimizer.state_dict(),
            'steps': steps,
            'epoch': epoch,
            'condition_vocab': condition_vocab.state_dict(),
            'val_loss': val_metrics['loss'],
            'val_ppl': val_metrics['ppl'],
            'val_acc': val_metrics['acc'],
            'noise_config_ids': noise_config_ids,
            'direction': direction,
        }

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(checkpoint_path / 'model_best.pt', {
                'model': transformer.state_dict(),
                'condition_vocab': condition_vocab.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_ppl': val_metrics['ppl'],
                'val_acc': val_metrics['acc'],
                'epoch': epoch,
                'noise_config_ids': noise_config_ids,
                'direction': direction,
            })
            best_message = (
                'Updated best checkpoint (val_loss={:.3f}) at epoch {}'
            ).format(best_val_loss, epoch + 1)
            tqdm.write(best_message)
            logger.info(best_message)

        save_checkpoint(checkpoint_path / 'model_latest.pt', checkpoint_payload)
        checkpoint_message = (
            'Saved latest checkpoint at end of epoch {} (step {})'
        ).format(epoch + 1, steps)
        tqdm.write(checkpoint_message)
        logger.info(checkpoint_message)

        epoch_time = int(time.time() - start)
        logger.info('Time taken for epoch %d is %d sec', epoch + 1, epoch_time)

    sw.close()


if __name__ == '__main__':
    run_training()
