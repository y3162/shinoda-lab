import logging
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from pesq import pesq
from torch.distributed import barrier
from tqdm import tqdm


DEFAULT_DIST_TIMEOUT_MINUTES = 120


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    return torch.load(filepath, map_location=device)


def save_checkpoint(filepath, obj):
    torch.save(obj, filepath)


def cal_pesq(clean, noisy, sr=16000):
    try:
        score = pesq(sr, clean, noisy, 'wb')
    except Exception:
        score = -1
    return score


def batch_pesq(clean, noisy):
    scores = np.array([
        cal_pesq(clean_utt, noisy_utt)
        for clean_utt, noisy_utt in zip(clean, noisy)
    ])
    if -1 in scores:
        return None
    scores = (scores - 1) / 3.5
    return torch.FloatTensor(scores)


def device_barrier(device, world_size):
    if world_size > 1:
        barrier(device_ids=[device.index])


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
    local_rank = (
        0
        if os.environ.get('SE_MAMBA_ISOLATED_DEVICE') == '1'
        else int(os.environ.get('LOCAL_RANK', 0))
    )
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, local_rank, world_size


def resolve_dist_timeout_minutes(h, override_minutes=None):
    if override_minutes is not None:
        return override_minutes
    return h.dist_config.get('dist_timeout_minutes', DEFAULT_DIST_TIMEOUT_MINUTES)


def setup_logging(log_dir, rank=0):
    logger = logging.getLogger('se_mamba_pp.train')
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if rank != 0:
        logger.addHandler(logging.NullHandler())
        return logger

    os.makedirs(log_dir, exist_ok=True)
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    file_handler = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def format_postfix(**kwargs):
    formatted = {}
    for key, value in kwargs.items():
        if isinstance(value, float):
            formatted[key] = f'{value:.3f}'
        else:
            formatted[key] = value
    return formatted


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
    return TorchrunTqdm(
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
    )


def resolve_checkpoint_path(checkpoint_root):
    if os.path.isdir(checkpoint_root):
        cp_g = os.path.join(checkpoint_root, 'g_latest')
        cp_do = os.path.join(checkpoint_root, 'do_latest')
        if os.path.isfile(cp_g) and os.path.isfile(cp_do):
            return checkpoint_root
        if os.path.isfile(os.path.join(checkpoint_root, 'config.json')):
            return checkpoint_root

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    parent = checkpoint_root
    if os.path.isfile(os.path.join(checkpoint_root, 'config.json')):
        parent = os.path.dirname(checkpoint_root)
    return os.path.join(parent, timestamp)


def latest_checkpoint_paths(checkpoint_dir):
    cp_g = os.path.join(checkpoint_dir, 'g_latest')
    cp_do = os.path.join(checkpoint_dir, 'do_latest')
    if os.path.isfile(cp_g) and os.path.isfile(cp_do):
        return cp_g, cp_do
    return None, None


def save_latest_checkpoint(
    checkpoint_path,
    generator,
    mssbcqtd,
    mrd,
    optim_g,
    optim_d,
    steps,
    epoch,
    num_gpus,
):
    gen = generator.module if num_gpus > 1 else generator
    disc_q = mssbcqtd.module if num_gpus > 1 else mssbcqtd
    disc_r = mrd.module if num_gpus > 1 else mrd
    save_checkpoint(
        os.path.join(checkpoint_path, 'g_latest'),
        {'generator': gen.state_dict()},
    )
    save_checkpoint(
        os.path.join(checkpoint_path, 'do_latest'),
        {
            'mssbcqtd': disc_q.state_dict(),
            'mrd': disc_r.state_dict(),
            'optim_g': optim_g.state_dict(),
            'optim_d': optim_d.state_dict(),
            'steps': steps,
            'epoch': epoch,
        },
    )


def save_best_checkpoint(checkpoint_path, generator, num_gpus):
    gen = generator.module if num_gpus > 1 else generator
    save_checkpoint(
        os.path.join(checkpoint_path, 'g_best'),
        {'generator': gen.state_dict()},
    )


def build_env(config, config_name, path):
    target_path = os.path.join(path, config_name)
    if config != target_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, target_path)


def aggregate_sum(value, device, world_size):
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()
