import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm


def override_json_with_args(json_path, args):
    with open(json_path, 'r') as f:
        config = json.load(f)
    return __json_to_namespace(__override_config(config, args))


def json_to_namespace(json_path):
    with open(json_path, 'r') as f:
        return __json_to_namespace(json.load(f))


def namespace_to_dict(value):
    if isinstance(value, argparse.Namespace):
        return {key: namespace_to_dict(val) for key, val in vars(value).items()}
    if isinstance(value, list):
        return [namespace_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: namespace_to_dict(val) for key, val in value.items()}
    return value


def apply_overrides(config, args):
    return __json_to_namespace(
        __override_config(namespace_to_dict(config), args),
    )


def __json_to_namespace(value):
    if isinstance(value, dict):
        return argparse.Namespace(**{
            key: __json_to_namespace(val) for key, val in value.items()
        })
    if isinstance(value, list):
        return [__json_to_namespace(item) for item in value]
    return value


def __override_config(config, args):
    for dotted_key, value in vars(args).items():
        if value is None or dotted_key in {
            'config', 'resume', 'checkpoint_root',
        }:
            continue
        keys = dotted_key.split('.')
        target = config
        for key in keys[:-1]:
            if key not in target:
                raise KeyError(f'Unknown config key: {dotted_key}')
            target = target[key]
        if keys[-1] not in target:
            raise KeyError(f'Unknown config key: {dotted_key}')
        target[keys[-1]] = value
    return config


def load_checkpoint(filepath, device):
    return torch.load(filepath, map_location=device, weights_only=False)


def save_checkpoint(filepath, obj):
    torch.save(obj, filepath)


def configure_runtime():
    os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
    os.environ.setdefault('NCCL_IB_DISABLE', '1')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def worker_init_fn(_worker_id):
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


def resolve_dist_info():
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, local_rank, world_size


def aggregate_sum(value, device, world_size):
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def format_postfix(**kwargs):
    return {
        key: f'{value:.3f}' if isinstance(value, float) else value
        for key, value in kwargs.items()
    }


class TorchrunTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('file', sys.stderr)
        kwargs.setdefault('dynamic_ncols', True)
        kwargs.setdefault('mininterval', 0.5)
        kwargs.setdefault('disable', False)
        super().__init__(*args, **kwargs)

    def display(self, msg=None, pos=None):
        if self.disable:
            return
        if msg is None:
            msg = self.__str__()
        if self.fp.isatty():
            return super().display(msg=msg, pos=pos)
        self.fp.write(msg + '\n')
        self.fp.flush()


def make_progress_bar(total, desc, unit='sample', leave=True):
    return TorchrunTqdm(total=total, desc=desc, unit=unit, leave=leave)


def unwrap(model, world_size):
    return model.module if world_size > 1 else model


def save_latest_checkpoint(
    run_dir,
    model,
    optim,
    scheduler,
    steps,
    epoch,
    best_val_loss,
    world_size,
):
    run_dir = Path(run_dir)
    save_checkpoint(
        run_dir / 'latest.pt',
        {
            'model': unwrap(model, world_size).state_dict(),
            'optim': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'steps': steps,
            'epoch': epoch,
            'best_val_loss': best_val_loss,
        },
    )


def save_best_checkpoint(
    run_dir,
    model,
    optim,
    scheduler,
    steps,
    epoch,
    best_val_loss,
    world_size,
):
    save_checkpoint(
        Path(run_dir) / 'best.pt',
        {
            'model': unwrap(model, world_size).state_dict(),
            'optim': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'steps': steps,
            'epoch': epoch,
            'best_val_loss': best_val_loss,
        },
    )


def set_warmup_lr(optim, base_lr, step, warmup_steps):
    if step >= warmup_steps:
        return
    scale = float(step + 1) / float(warmup_steps)
    for group in optim.param_groups:
        group['lr'] = base_lr * scale
