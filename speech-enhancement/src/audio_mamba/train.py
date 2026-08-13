import argparse
import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path


def _isolate_cuda_visible_device():
    if 'LOCAL_RANK' not in os.environ:
        return False
    local_rank = int(os.environ['LOCAL_RANK'])
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible:
        devices = [d.strip() for d in visible.split(',') if d.strip()]
        os.environ['CUDA_VISIBLE_DEVICES'] = devices[local_rank]
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
    return True


_CUDA_ISOLATED = _isolate_cuda_visible_device()

import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.audio_mamba.dataset import build_datasets
from src.audio_mamba.model.audiomamba import AudioMamba
from src.audio_mamba.model.loss import AudioMambaLoss
from src.audio_mamba.utils import (
    aggregate_sum,
    apply_overrides,
    configure_runtime,
    format_postfix,
    json_to_namespace,
    load_checkpoint,
    make_progress_bar,
    namespace_to_dict,
    override_json_with_args,
    resolve_dist_info,
    save_best_checkpoint,
    save_latest_checkpoint,
    set_warmup_lr,
    worker_init_fn,
)
from src.utils.print import print_log

torch.backends.cudnn.benchmark = True


def _device_index(local_rank):
    return 0 if _CUDA_ISOLATED else local_rank


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/audio_mamba/configs/default.json')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--checkpoint_root', default=None)
    for name, typ in [
        ('data.dataset', str),
        ('data.sampling_rate', int),
        ('data.segment_size', int),
        ('data.mel.n_fft', int),
        ('data.mel.hop_size', int),
        ('data.mel.win_size', int),
        ('data.mel.n_mels', int),
        ('data.mel.f_min', float),
        ('data.mel.f_max', float),
        ('data.mel.eps', float),
        ('data.specaug.time_mask', int),
        ('data.specaug.freq_mask', int),
        ('data.librispeech.sql_root', str),
        ('model.embed_dim', int),
        ('model.depth', int),
        ('model.d_state', int),
        ('model.d_conv', int),
        ('model.expand', int),
        ('model.norm_epsilon', float),
        ('model.drop_rate', float),
        ('model.num_classes', int),
        ('model.snr_loss_weight', float),
        ('train.env.batch_size', int),
        ('train.env.seed', int),
        ('train.env.num_workers', int),
        ('train.env.prefetch_factor', int),
        ('train.env.epochs', int),
        ('train.env.summary_interval', int),
        ('train.env.max_steps', int),
        ('train.env.val_batch_size', int),
        ('train.env.warmup_steps', int),
        ('train.optim.learning_rate', float),
        ('train.optim.adam_b1', float),
        ('train.optim.adam_b2', float),
        ('train.optim.weight_decay', float),
        ('train.optim.lr_gamma', float),
    ]:
        parser.add_argument(f'--{name}', type=typ, default=None)
    parser.add_argument('--data.librispeech.train_splits', nargs='+', default=None)
    parser.add_argument('--data.librispeech.validation_splits', nargs='+', default=None)
    parser.add_argument(
        '--data.librispeech.noise_config_ids', nargs='+', type=int, default=None,
    )
    parser.add_argument(
        '--train.optim.lr_milestones', nargs='+', type=int, default=None,
    )
    args = parser.parse_args()

    if args.resume is not None:
        print_log(f'Resuming from {args.resume}')
        config = apply_overrides(
            json_to_namespace(str(Path(args.resume) / 'config.json')),
            args,
        )
    else:
        config = override_json_with_args(args.config, args)

    if (
        config.data.dataset == 'librispeech'
        and config.data.librispeech.sql_root is None
    ):
        from src.config import SQL_ROOT
        config.data.librispeech.sql_root = str(SQL_ROOT)

    return config, args


def prepare_run(config, args):
    if args.checkpoint_root is not None:
        parent = Path(args.checkpoint_root)
    elif args.resume is not None:
        parent = Path(args.resume).parent
    else:
        parent = Path(config.checkpoint_root)

    run_dir = parent / datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.resume is not None:
        resume_dir = Path(args.resume)
        for name in ('latest.pt', 'best.pt'):
            src = resume_dir / name
            if src.is_file():
                shutil.copy2(src, run_dir / name)
        logs_src = resume_dir / 'logs'
        if logs_src.is_dir():
            shutil.copytree(logs_src, run_dir / 'logs')

    config.checkpoint_root = str(run_dir)
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(namespace_to_dict(config), f, indent=4)
    print_log(f'checkpoints directory: {run_dir}')
    return config


def build_model_and_optim(config, device):
    optim_cfg = config.train.optim
    model = AudioMamba(config.model).to(device)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=optim_cfg.learning_rate,
        betas=(optim_cfg.adam_b1, optim_cfg.adam_b2),
        weight_decay=optim_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optim,
        milestones=list(optim_cfg.lr_milestones),
        gamma=optim_cfg.lr_gamma,
    )
    loss_fn = AudioMambaLoss(config.model)
    return model, optim, scheduler, loss_fn


def compute_batch_metrics(output, noise_type, snr_db):
    preds = output.noise_type_logits.argmax(dim=1)
    accuracy = (preds == noise_type).float().mean().item()
    snr_mae = (output.snr_db.detach().view(-1) - snr_db.view(-1)).abs().mean().item()
    return accuracy, snr_mae


def validate(model, loss_fn, validset, device, config, rank, world_size, epoch):
    model.eval()
    torch.cuda.empty_cache()
    env = config.train.env
    val_batch_size = max(1, int(getattr(env, 'val_batch_size', env.batch_size)))
    valid_sampler = (
        DistributedSampler(validset, shuffle=False, drop_last=False)
        if world_size > 1 else None
    )
    loader = DataLoader(
        validset,
        num_workers=0,
        shuffle=False,
        sampler=valid_sampler,
        batch_size=val_batch_size,
        pin_memory=True,
        drop_last=False,
    )
    totals = {
        'loss': 0.0,
        'noise_type_loss': 0.0,
        'snr_loss': 0.0,
        'accuracy': 0.0,
        'snr_mae': 0.0,
        'n': 0,
    }
    pbar = None
    if rank == 0:
        pbar = make_progress_bar(
            total=len(loader),
            desc='Validation {}/{}'.format(epoch + 1, env.epochs),
            unit='batch',
            leave=False,
        )

    with torch.no_grad():
        for mel, noise_type, snr_db in loader:
            mel = mel.to(device, non_blocking=True)
            noise_type = noise_type.to(device, non_blocking=True)
            snr_db = snr_db.to(device, non_blocking=True)

            output = model(mel)
            losses = loss_fn(output, noise_type, snr_db)
            accuracy, snr_mae = compute_batch_metrics(output, noise_type, snr_db)

            batch_n = mel.size(0)
            totals['loss'] += losses['loss'].item() * batch_n
            totals['noise_type_loss'] += losses['noise_type_loss'].item() * batch_n
            totals['snr_loss'] += losses['snr_loss'].item() * batch_n
            totals['accuracy'] += accuracy * batch_n
            totals['snr_mae'] += snr_mae * batch_n
            totals['n'] += batch_n

            if pbar is not None:
                pbar.set_postfix(format_postfix(
                    loss=losses['loss'].item(),
                    acc=accuracy,
                    snr_mae=snr_mae,
                ), refresh=False)
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    n = aggregate_sum(totals['n'], device, world_size)
    return {
        'loss': aggregate_sum(totals['loss'], device, world_size) / max(n, 1),
        'noise_type_loss': aggregate_sum(totals['noise_type_loss'], device, world_size) / max(n, 1),
        'snr_loss': aggregate_sum(totals['snr_loss'], device, world_size) / max(n, 1),
        'accuracy': aggregate_sum(totals['accuracy'], device, world_size) / max(n, 1),
        'snr_mae': aggregate_sum(totals['snr_mae'], device, world_size) / max(n, 1),
    }


def train(config):
    rank, local_rank, world_size = resolve_dist_info()
    device_index = _device_index(local_rank)
    device = torch.device('cuda', device_index)
    torch.cuda.set_device(device_index)
    env = config.train.env
    optim_cfg = config.train.optim
    batch_size = max(1, int(env.batch_size // world_size))
    torch.random.default_generator.manual_seed(env.seed)
    torch.cuda.manual_seed(env.seed)

    steps = 0
    last_epoch = -1
    best_val_loss = float('inf')
    latest_path = Path(config.checkpoint_root) / 'latest.pt'
    state = None
    if latest_path.is_file():
        state = load_checkpoint(latest_path, device)
        steps = int(state.get('steps', 0)) + 1
        last_epoch = int(state.get('epoch', -1))
        best_val_loss = float(state.get('best_val_loss', float('inf')))
        if rank == 0:
            print_log(
                f"Loaded checkpoint (step {state.get('steps', 0)}, "
                f"epoch {state.get('epoch', -1) + 1}, "
                f"best_val_loss={best_val_loss:.4f})",
            )

    model, optim, scheduler, loss_fn = build_model_and_optim(config, device)
    if state is not None:
        model.load_state_dict(state['model'])
        optim.load_state_dict(state['optim'])
        scheduler.load_state_dict(state['scheduler'])

    if world_size > 1:
        ddp_kwargs = {
            'device_ids': [device_index],
            'output_device': device_index,
            'broadcast_buffers': False,
            'static_graph': True,
        }
        model = DistributedDataParallel(model, **ddp_kwargs)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print_log(f'Total Parameters: {n_params / 1e6:.3f}M')
        print_log(f'Batch size per GPU: {batch_size}')
        print_log(f'world_size: {world_size}')
        (Path(config.checkpoint_root) / 'logs').mkdir(parents=True, exist_ok=True)

    trainset, validset = build_datasets(config)
    train_sampler = DistributedSampler(trainset) if world_size > 1 else None
    loader_kwargs = {}
    if env.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = env.prefetch_factor
        loader_kwargs['worker_init_fn'] = worker_init_fn
    train_loader = DataLoader(
        trainset,
        num_workers=env.num_workers,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        **loader_kwargs,
    )

    sw = SummaryWriter(str(Path(config.checkpoint_root) / 'logs')) if rank == 0 else None
    start_epoch = 0 if last_epoch < 0 else last_epoch + 1
    stop = False
    warmup_steps = int(env.warmup_steps)
    base_lr = float(optim_cfg.learning_rate)

    model.train()
    for epoch in range(start_epoch, env.epochs):
        if rank == 0:
            print_log(f'Epoch: {epoch + 1}')
        if world_size > 1:
            train_sampler.set_epoch(epoch)

        train_pbar = make_progress_bar(
            total=len(train_loader) * batch_size,
            desc='Epoch {}/{}'.format(epoch + 1, env.epochs),
            unit='sample',
        ) if rank == 0 else None

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_batches = 0

        for mel, noise_type, snr_db in train_loader:
            set_warmup_lr(optim, base_lr, steps, warmup_steps)

            mel = mel.to(device, non_blocking=True)
            noise_type = noise_type.to(device, non_blocking=True)
            snr_db = snr_db.to(device, non_blocking=True)

            optim.zero_grad()
            output = model(mel)
            losses = loss_fn(output, noise_type, snr_db)
            losses['loss'].backward()
            optim.step()

            accuracy, snr_mae = compute_batch_metrics(output, noise_type, snr_db)
            epoch_loss += losses['loss'].item()
            epoch_acc += accuracy
            epoch_batches += 1

            if rank == 0:
                train_pbar.update(batch_size)
                train_pbar.set_postfix(format_postfix(
                    step=steps + 1,
                    loss=losses['loss'].item(),
                    acc=accuracy,
                    snr_mae=snr_mae,
                ), refresh=False)
                if steps % env.summary_interval == 0:
                    sw.add_scalar('Training/Loss', losses['loss'].item(), steps)
                    sw.add_scalar('Training/NoiseTypeLoss', losses['noise_type_loss'].item(), steps)
                    sw.add_scalar('Training/SNRLoss', losses['snr_loss'].item(), steps)
                    sw.add_scalar('Training/Accuracy', accuracy, steps)
                    sw.add_scalar('Training/SNR_MAE', snr_mae, steps)
                    sw.add_scalar('Training/LR', optim.param_groups[0]['lr'], steps)

            steps += 1
            if env.max_steps is not None and steps >= env.max_steps:
                stop = True
                break

        if rank == 0 and train_pbar is not None:
            train_pbar.close()
            if epoch_batches > 0:
                print_log(
                    f'Train epoch {epoch + 1}: '
                    f'loss={epoch_loss / epoch_batches:.4f}, '
                    f'acc={epoch_acc / epoch_batches:.4f}',
                )

        if stop:
            if rank == 0:
                save_latest_checkpoint(
                    config.checkpoint_root,
                    model, optim, scheduler,
                    steps, epoch - 1, best_val_loss, world_size,
                )
                print_log(f'Stopped at max_steps={env.max_steps} (step {steps})')
            break

        val_metrics = validate(
            model, loss_fn, validset, device, config, rank, world_size, epoch,
        )
        if rank == 0:
            msg = (
                'Validation (epoch {}/{}): loss={:.4f}, acc={:.4f}, '
                'snr_mae={:.4f}, noise_type_loss={:.4f}, snr_loss={:.4f}'
            ).format(
                epoch + 1, env.epochs,
                val_metrics['loss'], val_metrics['accuracy'], val_metrics['snr_mae'],
                val_metrics['noise_type_loss'], val_metrics['snr_loss'],
            )
            tqdm.write(msg)
            print_log(msg)
            sw.add_scalar('Validation/Loss', val_metrics['loss'], epoch + 1)
            sw.add_scalar('Validation/Accuracy', val_metrics['accuracy'], epoch + 1)
            sw.add_scalar('Validation/SNR_MAE', val_metrics['snr_mae'], epoch + 1)
            sw.add_scalar('Validation/NoiseTypeLoss', val_metrics['noise_type_loss'], epoch + 1)
            sw.add_scalar('Validation/SNRLoss', val_metrics['snr_loss'], epoch + 1)

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_best_checkpoint(
                    config.checkpoint_root,
                    model, optim, scheduler,
                    steps, epoch, best_val_loss, world_size,
                )
                print_log(
                    f'Updated best checkpoint (val_loss={best_val_loss:.4f}) at epoch {epoch + 1}',
                )
            save_latest_checkpoint(
                config.checkpoint_root,
                model, optim, scheduler,
                steps, epoch, best_val_loss, world_size,
            )
            print_log(f'Saved latest checkpoint at end of epoch {epoch + 1} (step {steps})')

        model.train()
        if world_size > 1:
            dist.barrier()
        if steps >= warmup_steps:
            scheduler.step()

    if sw is not None:
        sw.close()
    if world_size > 1:
        destroy_process_group()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for training.')
    configure_runtime()
    rank, local_rank, world_size = resolve_dist_info()
    device_index = _device_index(local_rank)
    torch.cuda.set_device(device_index)

    if world_size > 1:
        init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=120),
            device_id=torch.device('cuda', device_index),
        )

    config, args = parse_args()
    if rank == 0:
        config = prepare_run(config, args)
        objects = [config.checkpoint_root]
    else:
        objects = [None]
    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)
    config.checkpoint_root = objects[0]
    train(config)


if __name__ == '__main__':
    main()
