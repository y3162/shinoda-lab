import argparse
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.mp_senet.dataset import build_datasets
from src.mp_senet.model.discriminator import MetricDiscriminator
from src.mp_senet.model.model import MPNet, phase_losses
from src.mp_senet.utils import (
    aggregate_sum,
    apply_overrides,
    cal_pesq,
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
)
from src.utils.print import print_log

torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/mp_senet/configs/conformer.json')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--checkpoint_root', default=None)
    for name, typ in [
        ('data.dataset', str),
        ('data.sampling_rate', int),
        ('data.segment_size', int),
        ('data.stft.n_fft', int),
        ('data.stft.hop_size', int),
        ('data.stft.win_size', int),
        ('data.stft.compress_factor', float),
        ('data.voicebank.clean_train_dir', str),
        ('data.voicebank.noisy_train_dir', str),
        ('data.voicebank.clean_valid_dir', str),
        ('data.voicebank.noisy_valid_dir', str),
        ('data.voicebank.train_file', str),
        ('data.voicebank.valid_file', str),
        ('data.librispeech.sql_root', str),
        ('model.dense_channel', int),
        ('model.beta', float),
        ('model.bridge_block_type', str),
        ('model.num_tsblocks', int),
        ('model.n_heads', int),
        ('model.ffm_mult', int),
        ('model.ccm_expansion_factor', int),
        ('model.ccm_kernel_size', int),
        ('train.env.batch_size', int),
        ('train.env.seed', int),
        ('train.env.num_workers', int),
        ('train.env.epochs', int),
        ('train.env.summary_interval', int),
        ('train.env.max_steps', int),
        ('train.optim.learning_rate', float),
        ('train.optim.adam_b1', float),
        ('train.optim.adam_b2', float),
        ('train.optim.lr_decay', float),
        ('train.loss.magnitude', float),
        ('train.loss.phase', float),
        ('train.loss.complex', float),
        ('train.loss.stft', float),
        ('train.loss.metric', float),
        ('train.loss.time', float),
    ]:
        parser.add_argument(f'--{name}', type=typ, default=None)
    parser.add_argument('--data.librispeech.train_splits', nargs='+', default=None)
    parser.add_argument('--data.librispeech.validation_splits', nargs='+', default=None)
    parser.add_argument('--data.librispeech.noise_config_ids', nargs='+', type=int, default=None)
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
        for name in ('g_latest', 'do_latest', 'g_best'):
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


def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode='reflect',
        normalized=False,
        return_complex=True,
    )
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1) + 1e-9)
    pha = torch.atan2(
        stft_spec[:, :, :, 1] + 1e-10,
        stft_spec[:, :, :, 0] + 1e-5,
    )
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    return mag, pha, com


def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    mag = torch.pow(mag, 1.0 / compress_factor)
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    return torch.istft(
        com,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
    )


def build_models(config, device):
    optim = config.train.optim
    generator = MPNet(config.model, config.data.stft.n_fft).to(device)
    discriminator = MetricDiscriminator().to(device)
    optim_g = torch.optim.AdamW(
        generator.parameters(),
        optim.learning_rate,
        betas=[optim.adam_b1, optim.adam_b2],
    )
    optim_d = torch.optim.AdamW(
        discriminator.parameters(),
        optim.learning_rate,
        betas=[optim.adam_b1, optim.adam_b2],
    )
    return generator, discriminator, optim_g, optim_d


def forward_batch(generator, clean_audio, noisy_audio, config):
    stft = config.data.stft
    clean_mag, clean_pha, clean_com = mag_pha_stft(
        clean_audio, stft.n_fft, stft.hop_size, stft.win_size, stft.compress_factor,
    )
    noisy_mag, noisy_pha, _ = mag_pha_stft(
        noisy_audio, stft.n_fft, stft.hop_size, stft.win_size, stft.compress_factor,
    )
    mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)
    audio_g = mag_pha_istft(
        mag_g, pha_g, stft.n_fft, stft.hop_size, stft.win_size, stft.compress_factor,
    )
    mag_g_hat, _, com_g_hat = mag_pha_stft(
        audio_g, stft.n_fft, stft.hop_size, stft.win_size, stft.compress_factor,
    )
    loss_mag = F.mse_loss(clean_mag, mag_g)
    loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g)
    loss_pha = loss_ip + loss_gd + loss_iaf
    loss_com = F.mse_loss(clean_com, com_g) * 2
    loss_stft = F.mse_loss(com_g, com_g_hat) * 2
    outputs = {
        'clean_mag': clean_mag,
        'mag_g': mag_g,
        'mag_g_hat': mag_g_hat,
        'audio_g': audio_g,
    }
    return outputs, loss_mag, loss_pha, loss_com, loss_stft


def validate(generator, validset, device, config, rank, world_size, epoch):
    generator.eval()
    torch.cuda.empty_cache()
    valid_sampler = (
        DistributedSampler(validset, shuffle=False, drop_last=False)
        if world_size > 1 else None
    )
    loader = DataLoader(
        validset,
        num_workers=max(1, config.train.env.num_workers // 2),
        shuffle=False,
        sampler=valid_sampler,
        batch_size=1,
        pin_memory=True,
        drop_last=False,
    )
    totals = {'mag': 0.0, 'pha': 0.0, 'com': 0.0, 'stft': 0.0, 'pesq': 0.0, 'n': 0, 'pesq_n': 0}
    pbar = None
    if rank == 0:
        pbar = make_progress_bar(
            total=len(loader),
            desc='Validation {}/{}'.format(epoch + 1, config.train.env.epochs),
            unit='sample',
            leave=False,
        )

    with torch.no_grad():
        for clean_audio, noisy_audio in loader:
            clean_audio = clean_audio.to(device, non_blocking=True)
            noisy_audio = noisy_audio.to(device, non_blocking=True)
            outputs, loss_mag, loss_pha, loss_com, loss_stft = forward_batch(
                generator, clean_audio, noisy_audio, config,
            )
            audio_g = outputs['audio_g']
            if audio_g.size(1) > clean_audio.size(1):
                audio_g = audio_g[:, : clean_audio.size(1)]
            elif audio_g.size(1) < clean_audio.size(1):
                clean_audio = clean_audio[:, : audio_g.size(1)]

            for ref, est in zip(
                torch.split(clean_audio, 1, dim=0),
                torch.split(audio_g, 1, dim=0),
            ):
                totals['pesq'] += cal_pesq(
                    ref.squeeze().cpu().numpy(),
                    est.squeeze().cpu().numpy(),
                    config.data.sampling_rate,
                )
                totals['pesq_n'] += 1

            totals['mag'] += loss_mag.item()
            totals['pha'] += loss_pha.item()
            totals['com'] += loss_com.item() / 2
            totals['stft'] += loss_stft.item() / 2
            totals['n'] += 1
            if pbar is not None:
                pbar.set_postfix(format_postfix(
                    mag=loss_mag.item(),
                    pha=loss_pha.item(),
                    com=loss_com.item() / 2,
                    stft=loss_stft.item() / 2,
                ), refresh=False)
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    n = aggregate_sum(totals['n'], device, world_size)
    pesq_n = aggregate_sum(totals['pesq_n'], device, world_size)
    return {
        'mag': aggregate_sum(totals['mag'], device, world_size) / max(n, 1),
        'pha': aggregate_sum(totals['pha'], device, world_size) / max(n, 1),
        'com': aggregate_sum(totals['com'], device, world_size) / max(n, 1),
        'stft': aggregate_sum(totals['stft'], device, world_size) / max(n, 1),
        'pesq': aggregate_sum(totals['pesq'], device, world_size) / max(pesq_n, 1),
    }


def train(config):
    rank, local_rank, world_size = resolve_dist_info()
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(local_rank)
    env = config.train.env
    batch_size = max(1, int(env.batch_size // world_size))
    torch.random.default_generator.manual_seed(env.seed)
    torch.cuda.manual_seed(env.seed)

    steps = 0
    last_epoch = -1
    best_pesq = 0.0
    do_path = Path(config.checkpoint_root) / 'do_latest'
    g_path = Path(config.checkpoint_root) / 'g_latest'
    state_dict_do = None
    state_dict_g = None
    if g_path.is_file() and do_path.is_file():
        state_dict_g = load_checkpoint(g_path, device)
        state_dict_do = load_checkpoint(do_path, device)
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
        best_pesq = float(state_dict_do.get('best_pesq', 0.0))
        if rank == 0:
            print_log(
                f"Loaded checkpoint (step {state_dict_do['steps']}, "
                f"epoch {state_dict_do['epoch'] + 1}, best_pesq={best_pesq:.3f})",
            )

    generator, discriminator, optim_g, optim_d = build_models(config, device)
    if state_dict_do is not None:
        generator.load_state_dict(state_dict_g['generator'])
        discriminator.load_state_dict(state_dict_do['discriminator'])
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config.train.optim.lr_decay, last_epoch=last_epoch,
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config.train.optim.lr_decay, last_epoch=last_epoch,
    )

    if world_size > 1:
        generator = DistributedDataParallel(
            generator, device_ids=[local_rank],
        )
        discriminator = DistributedDataParallel(
            discriminator, device_ids=[local_rank],
        )

    if rank == 0:
        n_params = sum(p.numel() for p in generator.parameters())
        print_log(f'Total Parameters: {n_params / 1e6:.3f}M')
        print_log(f'Batch size per GPU: {batch_size}')
        print_log(f'world_size: {world_size}')
        (Path(config.checkpoint_root) / 'logs').mkdir(parents=True, exist_ok=True)

    trainset, validset = build_datasets(config)
    train_sampler = DistributedSampler(trainset) if world_size > 1 else None
    train_loader = DataLoader(
        trainset,
        num_workers=env.num_workers,
        shuffle=False,
        sampler=train_sampler,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
    )

    sw = SummaryWriter(str(Path(config.checkpoint_root) / 'logs')) if rank == 0 else None
    loss_w = config.train.loss
    start_epoch = 0 if last_epoch < 0 else last_epoch + 1
    stop = False

    generator.train()
    discriminator.train()

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

        for clean_audio, noisy_audio in train_loader:
            clean_audio = clean_audio.to(device, non_blocking=True)
            noisy_audio = noisy_audio.to(device, non_blocking=True)
            one_labels = torch.ones(clean_audio.size(0), device=device)

            outputs, loss_mag, loss_pha, loss_com, loss_stft = forward_batch(
                generator, clean_audio, noisy_audio, config,
            )
            clean_mag = outputs['clean_mag']
            mag_g_hat = outputs['mag_g_hat']
            audio_g = outputs['audio_g']

            optim_d.zero_grad()
            metric_r = discriminator(clean_mag, clean_mag)
            metric_g = discriminator(clean_mag, mag_g_hat.detach())
            loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())
            loss_disc_g = 0
            loss_disc_all = loss_disc_r + loss_disc_g
            loss_disc_all.backward()
            optim_d.step()

            optim_g.zero_grad()
            loss_time = F.l1_loss(clean_audio, audio_g)
            metric_g = discriminator(clean_mag, mag_g_hat)
            loss_metric = F.mse_loss(metric_g.flatten(), one_labels)
            loss_gen_all = (
                loss_mag * loss_w.magnitude
                + loss_pha * loss_w.phase
                + loss_com * loss_w.complex
                + loss_stft * loss_w.stft
                + loss_metric * loss_w.metric
                + loss_time * loss_w.time
            )
            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                train_pbar.update(batch_size)
                train_pbar.set_postfix(format_postfix(
                    step=steps + 1,
                    gen=float(loss_gen_all),
                    disc=float(loss_disc_all),
                    metric=float(loss_metric),
                    mag=loss_mag.item(),
                    pha=loss_pha.item(),
                    com=loss_com.item() / 2,
                    time=loss_time.item(),
                    stft=loss_stft.item() / 2,
                ), refresh=False)
                if steps % env.summary_interval == 0:
                    sw.add_scalar('Training/Generator Loss', loss_gen_all, steps)
                    sw.add_scalar('Training/Discriminator Loss', loss_disc_all, steps)
                    sw.add_scalar('Training/Metric Loss', loss_metric, steps)
                    sw.add_scalar('Training/Magnitude Loss', loss_mag.item(), steps)
                    sw.add_scalar('Training/Phase Loss', loss_pha.item(), steps)
                    sw.add_scalar('Training/Complex Loss', loss_com.item() / 2, steps)
                    sw.add_scalar('Training/Time Loss', loss_time.item(), steps)
                    sw.add_scalar('Training/Consistency Loss', loss_stft.item() / 2, steps)

            steps += 1
            if env.max_steps is not None and steps >= env.max_steps:
                stop = True
                break

        if rank == 0 and train_pbar is not None:
            train_pbar.close()

        if stop:
            if rank == 0:
                save_latest_checkpoint(
                    config.checkpoint_root,
                    generator, discriminator, optim_g, optim_d,
                    steps, epoch - 1, best_pesq, world_size,
                )
                print_log(f'Stopped at max_steps={env.max_steps} (step {steps})')
            break

        val_metrics = validate(
            generator, validset, device, config, rank, world_size, epoch,
        )
        if rank == 0:
            msg = (
                'Validation (epoch {}/{}): PESQ={:.3f}, mag={:.3f}, '
                'pha={:.3f}, com={:.3f}, stft={:.3f}'
            ).format(
                epoch + 1, env.epochs,
                val_metrics['pesq'], val_metrics['mag'], val_metrics['pha'],
                val_metrics['com'], val_metrics['stft'],
            )
            tqdm.write(msg)
            print_log(msg)
            sw.add_scalar('Validation/PESQ Score', val_metrics['pesq'], epoch + 1)
            sw.add_scalar('Validation/Magnitude Loss', val_metrics['mag'], epoch + 1)
            sw.add_scalar('Validation/Phase Loss', val_metrics['pha'], epoch + 1)
            sw.add_scalar('Validation/Complex Loss', val_metrics['com'], epoch + 1)
            sw.add_scalar('Validation/Consistency Loss', val_metrics['stft'], epoch + 1)
            if val_metrics['pesq'] > best_pesq:
                best_pesq = val_metrics['pesq']
                save_best_checkpoint(config.checkpoint_root, generator, world_size)
                print_log(
                    f'Updated best checkpoint (PESQ={best_pesq:.3f}) at epoch {epoch + 1}',
                )
            save_latest_checkpoint(
                config.checkpoint_root,
                generator, discriminator, optim_g, optim_d,
                steps, epoch, best_pesq, world_size,
            )
            print_log(f'Saved latest checkpoint at end of epoch {epoch + 1} (step {steps})')

        generator.train()
        if world_size > 1:
            dist.barrier()
        scheduler_g.step()
        scheduler_d.step()

    if sw is not None:
        sw.close()
    if world_size > 1:
        destroy_process_group()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for training.')
    configure_runtime()
    rank, local_rank, world_size = resolve_dist_info()
    torch.cuda.set_device(local_rank)

    if world_size > 1:
        init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=120),
            device_id=torch.device('cuda', local_rank),
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
