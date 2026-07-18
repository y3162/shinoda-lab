import argparse
import itertools
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

from src.se_mamba_pp.dataset import build_datasets
from src.se_mamba_pp.model.discriminator import (
    MultiResolutionDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from src.se_mamba_pp.model.loss import MultiScaleMelSpectrogramLoss, phase_losses
from src.se_mamba_pp.model.semambapp import SEMambapp
from src.se_mamba_pp.model.stfts import forward_generator_batch
from src.se_mamba_pp.utils import (
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
    worker_init_fn,
)
from src.utils.print import print_log

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/se_mamba_pp/configs/default.json')
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
        ('model.hid_feature', int),
        ('model.unet_expansion', int),
        ('model.num_tfmamba', int),
        ('model.d_state', int),
        ('model.d_conv', int),
        ('model.expand', int),
        ('model.norm_epsilon', float),
        ('model.beta', float),
        ('model.input_channel', int),
        ('model.output_channel', int),
        ('model.dropout', float),
        ('train.env.batch_size', int),
        ('train.env.seed', int),
        ('train.env.num_workers', int),
        ('train.env.prefetch_factor', int),
        ('train.env.epochs', int),
        ('train.env.summary_interval', int),
        ('train.env.max_steps', int),
        ('train.optim.learning_rate', float),
        ('train.optim.adam_b1', float),
        ('train.optim.adam_b2', float),
        ('train.optim.lr_decay', float),
        ('train.loss.adv_g', float),
        ('train.loss.fm_g', float),
        ('train.loss.magnitude', float),
        ('train.loss.phase', float),
        ('train.loss.complex', float),
        ('train.loss.mel', float),
        ('train.loss.consistancy', float),
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


def build_models(config, device):
    optim = config.train.optim
    generator = SEMambapp(config.model, config.data.stft.n_fft).to(device)
    mssbcqtd = MultiScaleSubbandCQTDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)
    fn_mel_loss = MultiScaleMelSpectrogramLoss(
        sampling_rate=config.data.sampling_rate,
    ).to(device)

    adamw_kwargs = {
        'lr': optim.learning_rate,
        'betas': (optim.adam_b1, optim.adam_b2),
        'fused': True,
    }
    optim_g = torch.optim.AdamW(generator.parameters(), **adamw_kwargs)
    optim_d = torch.optim.AdamW(
        itertools.chain(mrd.parameters(), mssbcqtd.parameters()),
        **adamw_kwargs,
    )
    return generator, mssbcqtd, mrd, fn_mel_loss, optim_g, optim_d


def forward_batch(generator, clean_audio, noisy_audio, config):
    outputs = forward_generator_batch(
        generator, clean_audio, noisy_audio, config.data.stft,
    )
    loss_mag = F.mse_loss(outputs['clean_mag'], outputs['mag_g'])
    loss_ip, loss_gd, loss_iaf = phase_losses(
        outputs['clean_pha'], outputs['pha_g'], config.data.stft,
    )
    loss_pha = loss_ip + loss_gd + loss_iaf
    loss_com = F.mse_loss(outputs['clean_com'], outputs['com_g']) * 2
    loss_con = F.mse_loss(outputs['com_g'], outputs['rec_com']) * 2
    return outputs, loss_mag, loss_pha, loss_com, loss_con


def validate(generator, validset, device, config, rank, world_size, epoch):
    generator.eval()
    torch.cuda.empty_cache()
    valid_sampler = (
        DistributedSampler(validset, shuffle=False, drop_last=False)
        if world_size > 1 else None
    )
    loader = DataLoader(
        validset,
        num_workers=0,
        shuffle=False,
        sampler=valid_sampler,
        batch_size=1,
        pin_memory=True,
        drop_last=False,
    )
    totals = {'mag': 0.0, 'pha': 0.0, 'com': 0.0, 'con': 0.0, 'pesq': 0.0, 'n': 0, 'pesq_n': 0}
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
            outputs, loss_mag, loss_pha, loss_com, loss_con = forward_batch(
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
            totals['con'] += loss_con.item() / 2
            totals['n'] += 1
            if pbar is not None:
                pbar.set_postfix(format_postfix(
                    mag=loss_mag.item(),
                    pha=loss_pha.item(),
                    com=loss_com.item() / 2,
                    con=loss_con.item() / 2,
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
        'stft': aggregate_sum(totals['con'], device, world_size) / max(n, 1),
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

    generator, mssbcqtd, mrd, fn_mel_loss, optim_g, optim_d = build_models(config, device)
    if state_dict_do is not None:
        generator.load_state_dict(state_dict_g['generator'])
        mssbcqtd.load_state_dict(state_dict_do['mssbcqtd'])
        mrd.load_state_dict(state_dict_do['mrd'])
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config.train.optim.lr_decay, last_epoch=last_epoch,
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config.train.optim.lr_decay, last_epoch=last_epoch,
    )

    if world_size > 1:
        ddp_kwargs = {
            'device_ids': [local_rank],
            'output_device': local_rank,
            'broadcast_buffers': False,
            'static_graph': True,
        }
        generator = DistributedDataParallel(generator, **ddp_kwargs)
        mssbcqtd = DistributedDataParallel(mssbcqtd, **ddp_kwargs)
        mrd = DistributedDataParallel(mrd, **ddp_kwargs)

    if rank == 0:
        n_params = sum(p.numel() for p in generator.parameters())
        print_log(f'SEMambapp Parameters: {n_params / 1e6:.3f}M')
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
    loss_w = config.train.loss
    start_epoch = 0 if last_epoch < 0 else last_epoch + 1
    stop = False

    generator.train()
    mssbcqtd.train()
    mrd.train()

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
            outputs, loss_mag, loss_pha, loss_com, loss_con = forward_batch(
                generator, clean_audio, noisy_audio, config,
            )
            audio_g = outputs['audio_g']

            optim_d.zero_grad(set_to_none=True)
            y_dq_hat_r, y_dq_hat_g, _, _ = mssbcqtd(
                clean_audio.unsqueeze(1), audio_g.unsqueeze(1).detach(),
            )
            loss_disc_q, _, _ = discriminator_loss(y_dq_hat_r, y_dq_hat_g)
            y_dr_hat_r, y_dr_hat_g, _, _ = mrd(
                clean_audio.unsqueeze(1), audio_g.unsqueeze(1).detach(),
            )
            loss_disc_r, _, _ = discriminator_loss(y_dr_hat_r, y_dr_hat_g)
            loss_disc_all = loss_disc_q + loss_disc_r
            loss_disc_all.backward()
            optim_d.step()

            optim_g.zero_grad(set_to_none=True)
            y_dq_hat_r, y_dq_hat_g, fmap_q_r, fmap_q_g = mssbcqtd(
                clean_audio.unsqueeze(1), audio_g.unsqueeze(1),
            )
            y_dr_hat_r, y_dr_hat_g, fmap_r_r, fmap_r_g = mrd(
                clean_audio.unsqueeze(1), audio_g.unsqueeze(1),
            )
            adv_g_loss = (
                generator_loss(y_dq_hat_g)[0] + generator_loss(y_dr_hat_g)[0]
            )
            fm_g_loss = feature_loss(fmap_q_r, fmap_q_g) + feature_loss(fmap_r_r, fmap_r_g)
            mel_loss = fn_mel_loss(clean_audio.unsqueeze(1), audio_g.unsqueeze(1))
            loss_gen_all = (
                adv_g_loss * loss_w.adv_g
                + fm_g_loss * loss_w.fm_g
                + mel_loss * loss_w.mel
                + loss_mag * loss_w.magnitude
                + loss_pha * loss_w.phase
                + loss_com * loss_w.complex
                + loss_con * loss_w.consistancy
            )
            if torch.isnan(loss_gen_all).any():
                raise ValueError('NaN values found in loss_gen_all')
            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                train_pbar.update(batch_size)
                postfix = format_postfix(
                    step=steps + 1,
                    gen=loss_gen_all.item(),
                    disc=loss_disc_all.item(),
                    adv=adv_g_loss.item(),
                    fm=fm_g_loss.item(),
                    mag=loss_mag.item(),
                    pha=loss_pha.item(),
                    com=loss_com.item() / 2,
                    mel=mel_loss.item(),
                    con=loss_con.item() / 2,
                )
                train_pbar.set_postfix(postfix, refresh=True)
                if steps % env.summary_interval == 0:
                    sw.add_scalar('Training/Generator Loss', loss_gen_all.item(), steps)
                    sw.add_scalar('Training/Discriminator Loss', loss_disc_all.item(), steps)
                    sw.add_scalar('Training/adv_g_loss', adv_g_loss.item(), steps)
                    sw.add_scalar('Training/fm_g_loss', fm_g_loss.item(), steps)
                    sw.add_scalar('Training/Magnitude Loss', loss_mag.item(), steps)
                    sw.add_scalar('Training/Phase Loss', loss_pha.item(), steps)
                    sw.add_scalar('Training/Complex Loss', loss_com.item() / 2, steps)
                    sw.add_scalar('Training/Consistency Loss', loss_con.item() / 2, steps)
                    sw.add_scalar('Training/Mel Loss', mel_loss.item(), steps)
                    sw.add_scalar(
                        'Training/Time Loss',
                        F.l1_loss(clean_audio, audio_g).item(),
                        steps,
                    )

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
                    generator, mssbcqtd, mrd, optim_g, optim_d,
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
                generator, mssbcqtd, mrd, optim_g, optim_d,
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
