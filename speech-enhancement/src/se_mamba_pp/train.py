import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import itertools
import json
import logging
import os
import shutil
import time
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed import barrier, destroy_process_group, init_process_group
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
from src.se_mamba_pp.model.stfts import mag_phase_istft, mag_phase_stft
from src.se_mamba_pp.utils import AttrDict, cal_pesq, load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmark = True

CHECKPOINT_ROOT = 'data/checkpoints/se_mamba_pp'
DEFAULT_DIST_TIMEOUT_MINUTES = 120


def device_barrier(device, world_size):
    if world_size > 1:
        barrier(device_ids=[device.index])


def configure_distributed_runtime():
    os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
    os.environ.setdefault('NCCL_IB_DISABLE', '1')


def dist_init_method(checkpoint_path):
    init_file = os.path.abspath(os.path.join(checkpoint_path, '.dist_init'))
    if os.path.isfile(init_file):
        os.remove(init_file)
    return 'file://' + init_file


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
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))


def aggregate_sum(value, device, world_size):
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def forward_generator_batch(generator, clean_audio, noisy_audio, h):
    clean_mag, clean_pha, clean_com = mag_phase_stft(
        clean_audio,
        h.n_fft,
        h.hop_size,
        h.win_size,
        h.compress_factor,
    )
    noisy_mag, noisy_pha, _ = mag_phase_stft(
        noisy_audio,
        h.n_fft,
        h.hop_size,
        h.win_size,
        h.compress_factor,
    )
    mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)
    audio_g = mag_phase_istft(
        mag_g,
        pha_g,
        h.n_fft,
        h.hop_size,
        h.win_size,
        h.compress_factor,
    )
    _, _, rec_com = mag_phase_stft(
        audio_g,
        h.n_fft,
        h.hop_size,
        h.win_size,
        h.compress_factor,
        addeps=True,
    )
    return {
        'clean_mag': clean_mag,
        'clean_pha': clean_pha,
        'clean_com': clean_com,
        'noisy_mag': noisy_mag,
        'noisy_pha': noisy_pha,
        'mag_g': mag_g,
        'pha_g': pha_g,
        'com_g': com_g,
        'audio_g': audio_g,
        'rec_com': rec_com,
    }


def run_validation(
    generator,
    validation_loader,
    device,
    h,
    *,
    rank,
    world_size,
    epoch,
    total_epochs,
):
    generator.eval()
    torch.cuda.empty_cache()

    val_mag_err_tot = 0.0
    val_pha_err_tot = 0.0
    val_com_err_tot = 0.0
    val_con_err_tot = 0.0
    num_batches = 0
    pesq_sum = 0.0
    pesq_count = 0

    val_pbar = None
    if rank == 0:
        val_pbar = tqdm(
            total=len(validation_loader),
            unit='sample',
            desc='Validation {}/{}'.format(epoch + 1, total_epochs),
            dynamic_ncols=True,
            leave=False,
        )

    with torch.no_grad():
        for batch in validation_loader:
            clean_audio, noisy_audio = batch
            clean_audio = clean_audio.to(device, non_blocking=True)
            noisy_audio = noisy_audio.to(device, non_blocking=True)

            outputs = forward_generator_batch(
                generator,
                clean_audio,
                noisy_audio,
                h,
            )
            mag_error = F.mse_loss(
                outputs['clean_mag'],
                outputs['mag_g'],
            ).item()
            ip_error, gd_error, iaf_error = phase_losses(
                outputs['clean_pha'],
                outputs['pha_g'],
                h,
            )
            pha_error = (ip_error + gd_error + iaf_error).item()
            com_error = F.mse_loss(
                outputs['clean_com'],
                outputs['com_g'],
            ).item()
            con_error = F.mse_loss(
                outputs['com_g'],
                outputs['rec_com'],
            ).item()

            audio_g = outputs['audio_g']
            if audio_g.size(1) > clean_audio.size(1):
                audio_g = audio_g[:, : clean_audio.size(1)]
            elif audio_g.size(1) < clean_audio.size(1):
                clean_audio = clean_audio[:, : audio_g.size(1)]

            for ref, est in zip(
                torch.split(clean_audio, 1, dim=0),
                torch.split(audio_g, 1, dim=0),
            ):
                pesq_sum += cal_pesq(
                    ref.squeeze().cpu().numpy(),
                    est.squeeze().cpu().numpy(),
                    h.sampling_rate,
                )
                pesq_count += 1

            val_mag_err_tot += mag_error
            val_pha_err_tot += pha_error
            val_com_err_tot += com_error
            val_con_err_tot += con_error
            num_batches += 1

            if val_pbar is not None:
                val_pbar.set_postfix(
                    format_postfix(
                        mag=mag_error,
                        pha=pha_error,
                        com=com_error,
                        con=con_error,
                    ),
                    refresh=False,
                )
                val_pbar.update(1)

    if val_pbar is not None:
        val_pbar.close()

    total_batches = aggregate_sum(num_batches, device, world_size)
    val_mag_err = aggregate_sum(val_mag_err_tot, device, world_size) / total_batches
    val_pha_err = aggregate_sum(val_pha_err_tot, device, world_size) / total_batches
    val_com_err = aggregate_sum(val_com_err_tot, device, world_size) / total_batches
    val_con_err = aggregate_sum(val_con_err_tot, device, world_size) / total_batches
    total_pesq_sum = aggregate_sum(pesq_sum, device, world_size)
    total_pesq_count = aggregate_sum(pesq_count, device, world_size)
    val_pesq_score = (
        total_pesq_sum / total_pesq_count if total_pesq_count > 0 else 0.0
    )

    return {
        'pesq': val_pesq_score,
        'mag': val_mag_err,
        'pha': val_pha_err,
        'com': val_com_err,
        'stft': val_con_err,
    }


def train(rank, a, h):
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    device = torch.device('cuda:{:d}'.format(rank))

    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_init_method,
            world_size=h.dist_config['world_size'] * h.num_gpus,
            rank=rank,
            timeout=timedelta(minutes=h.dist_timeout_minutes),
            device_id=device,
        )

    logger = setup_logging(a.checkpoint_path, rank=rank)
    torch.cuda.manual_seed(h.seed)

    generator = SEMambapp(h).to(device)
    mssbcqtd = MultiScaleSubbandCQTDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)
    fn_mel_loss = MultiScaleMelSpectrogramLoss(
        sampling_rate=h.sampling_rate,
    ).to(device)

    if rank == 0:
        num_params = sum(p.numel() for p in generator.parameters())
        logger.info('SEMambapp Parameters: {:.3f}M'.format(num_params / 1e6))
        os.makedirs(a.checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(a.checkpoint_path, 'logs'), exist_ok=True)
        logger.info('checkpoints directory: %s', a.checkpoint_path)

    cp_g = None
    cp_do = None
    if os.path.isdir(a.checkpoint_path):
        cp_g, cp_do = latest_checkpoint_paths(a.checkpoint_path)

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
        if rank == 0:
            logger.info(
                'Starting training from scratch (g_latest/do_latest not found in %s)',
                a.checkpoint_path,
            )
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mssbcqtd.load_state_dict(state_dict_do['mssbcqtd'])
        mrd.load_state_dict(state_dict_do['mrd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
        if rank == 0:
            logger.info(
                'Resuming from step %d, epoch %d',
                state_dict_do['steps'],
                state_dict_do['epoch'] + 1,
            )

    if h.num_gpus > 1:
        device_barrier(device, h.num_gpus)
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mssbcqtd = DistributedDataParallel(mssbcqtd, device_ids=[rank]).to(device)
        mrd = DistributedDataParallel(mrd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(
        generator.parameters(),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(mrd.parameters(), mssbcqtd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g,
        gamma=h.lr_decay,
        last_epoch=last_epoch,
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d,
        gamma=h.lr_decay,
        last_epoch=last_epoch,
    )

    trainset, validset = build_datasets(a, h)
    world_size = h.num_gpus
    train_sampler = DistributedSampler(trainset) if world_size > 1 else None
    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=False,
        sampler=train_sampler,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True,
    )
    valid_sampler = (
        DistributedSampler(validset, shuffle=False, drop_last=False)
        if world_size > 1 else None
    )
    validation_loader = DataLoader(
        validset,
        num_workers=max(1, h.num_workers // 2),
        shuffle=False,
        sampler=valid_sampler,
        batch_size=1,
        pin_memory=True,
        drop_last=False,
    )

    sw = None
    if rank == 0:
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    mssbcqtd.train()
    mrd.train()

    best_pesq = 0
    start_epoch = 0 if last_epoch < 0 else last_epoch + 1

    for epoch in range(start_epoch, a.training_epochs):
        if rank == 0:
            start = time.time()
            logger.info('Epoch: %d', epoch + 1)

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            train_pbar = tqdm(
                total=len(train_loader) * h.batch_size,
                unit='sample',
                desc='Epoch {}/{}'.format(epoch + 1, a.training_epochs),
                dynamic_ncols=True,
            )
        else:
            train_pbar = None

        for batch in train_loader:
            clean_audio, noisy_audio = batch
            clean_audio = clean_audio.to(device, non_blocking=True)
            noisy_audio = noisy_audio.to(device, non_blocking=True)

            outputs = forward_generator_batch(
                generator,
                clean_audio,
                noisy_audio,
                h,
            )
            clean_mag = outputs['clean_mag']
            clean_pha = outputs['clean_pha']
            clean_com = outputs['clean_com']
            mag_g = outputs['mag_g']
            pha_g = outputs['pha_g']
            com_g = outputs['com_g']
            audio_g = outputs['audio_g']
            rec_com = outputs['rec_com']

            optim_d.zero_grad()
            y_dq_hat_r, y_dq_hat_g, _, _ = mssbcqtd(
                clean_audio.unsqueeze(1),
                audio_g.unsqueeze(1).detach(),
            )
            loss_disc_q, _, _ = discriminator_loss(y_dq_hat_r, y_dq_hat_g)
            y_dr_hat_r, y_dr_hat_g, _, _ = mrd(
                clean_audio.unsqueeze(1),
                audio_g.unsqueeze(1).detach(),
            )
            loss_disc_r, _, _ = discriminator_loss(y_dr_hat_r, y_dr_hat_g)
            loss_disc_all = loss_disc_q + loss_disc_r
            loss_disc_all.backward()
            optim_d.step()

            optim_g.zero_grad()
            y_dq_hat_r, y_dq_hat_g, fmap_q_r, fmap_q_g = mssbcqtd(
                clean_audio.unsqueeze(1),
                audio_g.unsqueeze(1),
            )
            loss_fm_q = feature_loss(fmap_q_r, fmap_q_g)
            loss_gen_q, _ = generator_loss(y_dq_hat_g)

            y_dr_hat_r, y_dr_hat_g, fmap_r_r, fmap_r_g = mrd(
                clean_audio.unsqueeze(1),
                audio_g.unsqueeze(1),
            )
            loss_fm_r = feature_loss(fmap_r_r, fmap_r_g)
            loss_gen_r, _ = generator_loss(y_dr_hat_g)

            adv_g_loss = loss_gen_q + loss_gen_r
            fm_g_loss = loss_fm_q + loss_fm_r
            loss_mag = F.mse_loss(clean_mag, mag_g)
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g, h)
            loss_pha = loss_ip + loss_gd + loss_iaf
            loss_com = F.mse_loss(clean_com, com_g) * 2
            loss_con = F.mse_loss(com_g, rec_com) * 2
            mel_loss = fn_mel_loss(
                clean_audio.unsqueeze(1),
                audio_g.unsqueeze(1),
            )

            loss_gen_all = (
                adv_g_loss * h.loss_adv_g
                + fm_g_loss * h.loss_fm_g
                + mel_loss * h.loss_mel
                + loss_mag * h.loss_magnitude
                + loss_pha * h.loss_phase
                + loss_com * h.loss_complex
                + loss_con * h.loss_consistancy
            )
            if torch.isnan(loss_gen_all).any():
                raise ValueError('NaN values found in loss_gen_all')
            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                with torch.no_grad():
                    mag_error = F.mse_loss(clean_mag, mag_g).item()
                    ip_error, gd_error, iaf_error = phase_losses(
                        clean_pha,
                        pha_g,
                        h,
                    )
                    pha_error = (ip_error + gd_error + iaf_error).item()
                    com_error = F.mse_loss(clean_com, com_g).item()
                    con_error = F.mse_loss(com_g, rec_com).item()
                    mel_error = mel_loss.item()
                    time_error = F.l1_loss(clean_audio, audio_g).item()

                if steps % a.summary_interval == 0:
                    sw.add_scalar('Training/Generator Loss', loss_gen_all, steps)
                    sw.add_scalar(
                        'Training/Discriminator Loss',
                        loss_disc_all,
                        steps,
                    )
                    sw.add_scalar('Training/adv_g_loss', float(adv_g_loss), steps)
                    sw.add_scalar('Training/fm_g_loss', float(fm_g_loss), steps)
                    sw.add_scalar('Training/Magnitude Loss', mag_error, steps)
                    sw.add_scalar('Training/Phase Loss', pha_error, steps)
                    sw.add_scalar('Training/Complex Loss', com_error, steps)
                    sw.add_scalar('Training/Consistency Loss', con_error, steps)
                    sw.add_scalar('Training/Mel Loss', mel_error, steps)
                    sw.add_scalar('Training/Time Loss', time_error, steps)

                train_pbar.set_postfix(
                    format_postfix(
                        step=steps + 1,
                        gen=float(loss_gen_all),
                        disc=float(loss_disc_all),
                        mag=mag_error,
                        pha=pha_error,
                        com=com_error,
                        mel=mel_error,
                    ),
                    refresh=False,
                )
                train_pbar.update(h.batch_size)

            steps += 1

        if rank == 0:
            train_pbar.close()

        val_metrics = run_validation(
            generator,
            validation_loader,
            device,
            h,
            rank=rank,
            world_size=world_size,
            epoch=epoch,
            total_epochs=a.training_epochs,
        )

        if rank == 0:
            val_message = (
                'Validation (epoch {}/{}): PESQ={:.3f}, mag={:.3f}, '
                'pha={:.3f}, com={:.3f}, stft={:.3f}'
            ).format(
                epoch + 1,
                a.training_epochs,
                val_metrics['pesq'],
                val_metrics['mag'],
                val_metrics['pha'],
                val_metrics['com'],
                val_metrics['stft'],
            )
            tqdm.write(val_message)
            logger.info(val_message)

            sw.add_scalar('Validation/PESQ Score', val_metrics['pesq'], epoch + 1)
            sw.add_scalar('Validation/Magnitude Loss', val_metrics['mag'], epoch + 1)
            sw.add_scalar('Validation/Phase Loss', val_metrics['pha'], epoch + 1)
            sw.add_scalar('Validation/Complex Loss', val_metrics['com'], epoch + 1)
            sw.add_scalar(
                'Validation/Consistency Loss',
                val_metrics['stft'],
                epoch + 1,
            )

            if val_metrics['pesq'] > best_pesq:
                best_pesq = val_metrics['pesq']
                save_best_checkpoint(a.checkpoint_path, generator, h.num_gpus)
                best_message = (
                    'Updated best checkpoint (PESQ={:.3f}) at epoch {}'
                ).format(best_pesq, epoch + 1)
                tqdm.write(best_message)
                logger.info(best_message)

            save_latest_checkpoint(
                a.checkpoint_path,
                generator,
                mssbcqtd,
                mrd,
                optim_g,
                optim_d,
                steps,
                epoch,
                h.num_gpus,
            )
            checkpoint_message = (
                'Saved latest checkpoint at end of epoch {} (step {})'
            ).format(epoch + 1, steps)
            tqdm.write(checkpoint_message)
            logger.info(checkpoint_message)

        generator.train()
        if h.num_gpus > 1:
            device_barrier(device, h.num_gpus)

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            logger.info(
                'Time taken for epoch %d is %d sec',
                epoch + 1,
                int(time.time() - start),
            )

    if h.num_gpus > 1:
        destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='voicebank',
        choices=['voicebank', 'librispeech'],
    )
    parser.add_argument(
        '--input_clean_wavs_dir',
        default='data/raw/VoiceBank+DEMAND/clean_trainset_56spk_wav',
    )
    parser.add_argument(
        '--input_noisy_wavs_dir',
        default='data/raw/VoiceBank+DEMAND/noisy_trainset_56spk_wav',
    )
    parser.add_argument(
        '--input_validation_clean_wavs_dir',
        default='data/raw/VoiceBank+DEMAND/clean_testset_wav',
    )
    parser.add_argument(
        '--input_validation_noisy_wavs_dir',
        default='data/raw/VoiceBank+DEMAND/noisy_testset_wav',
    )
    parser.add_argument(
        '--input_training_file',
        default='data/raw/VoiceBank+DEMAND/log_trainset_56spk.txt',
    )
    parser.add_argument(
        '--input_validation_file',
        default='data/raw/VoiceBank+DEMAND/log_testset.txt',
    )
    parser.add_argument('--sql_root', default=None)
    parser.add_argument('--train_splits', default=None, nargs='+')
    parser.add_argument('--validation_splits', default=None, nargs='+')
    parser.add_argument('--noise_config_ids', default=None, nargs='+', type=int)
    parser.add_argument('--checkpoint_path', default=CHECKPOINT_ROOT)
    parser.add_argument(
        '--config',
        default='src/se_mamba_pp/configs/default.json',
    )
    parser.add_argument('--training_epochs', default=100, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--num_workers', default=None, type=int)
    parser.add_argument('--dist_timeout_minutes', default=None, type=int)
    a = parser.parse_args()

    if a.dataset == 'librispeech':
        if a.sql_root is None:
            from src.config import SQL_ROOT
            a.sql_root = str(SQL_ROOT)
        if not a.train_splits:
            parser.error('--train_splits is required for --dataset librispeech')
        if not a.validation_splits:
            parser.error('--validation_splits is required for --dataset librispeech')

    a.checkpoint_path = resolve_checkpoint_path(a.checkpoint_path)
    os.makedirs(a.checkpoint_path, exist_ok=True)

    logger = setup_logging(a.checkpoint_path, rank=0)
    tqdm.write('Initializing Training Process..')
    tqdm.write('checkpoints directory: {}'.format(a.checkpoint_path))
    logger.info('Initializing Training Process..')
    logger.info('checkpoints directory: %s', a.checkpoint_path)

    with open(a.config) as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    if a.num_workers is not None:
        h.num_workers = a.num_workers
    h.dist_init_method = dist_init_method(a.checkpoint_path)
    h.dist_timeout_minutes = resolve_dist_timeout_minutes(
        h,
        override_minutes=a.dist_timeout_minutes,
    )
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for training.')

    torch.cuda.manual_seed(h.seed)
    h.num_gpus = torch.cuda.device_count()
    h.batch_size = int(h.batch_size / h.num_gpus)
    tqdm.write('Batch size per GPU: {}'.format(h.batch_size))
    tqdm.write('num_workers: {}'.format(h.num_workers))
    logger.info('Batch size per GPU: %d', h.batch_size)
    logger.info('num_workers: %d', h.num_workers)

    if h.num_gpus > 1:
        configure_distributed_runtime()
        tqdm.write('dist_init_method: {}'.format(h.dist_init_method))
        logger.info('dist_init_method: %s', h.dist_init_method)
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
