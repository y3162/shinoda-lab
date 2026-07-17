import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import itertools
import json
import os
import time
from datetime import timedelta

if 'LOCAL_RANK' in os.environ:
    local_rank = int(os.environ['LOCAL_RANK'])
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices.split(',')[local_rank]
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
    os.environ['SE_MAMBA_ISOLATED_DEVICE'] = '1'

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

if os.environ.get('SE_MAMBA_ISOLATED_DEVICE') == '1':
    torch.cuda.set_device(0)

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
    AttrDict,
    aggregate_sum,
    build_env,
    cal_pesq,
    configure_runtime,
    device_barrier,
    format_postfix,
    latest_checkpoint_paths,
    load_checkpoint,
    make_progress_bar,
    resolve_checkpoint_path,
    resolve_dist_info,
    resolve_dist_timeout_minutes,
    save_best_checkpoint,
    save_latest_checkpoint,
    setup_logging,
    worker_init_fn,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

CHECKPOINT_ROOT = 'data/checkpoints/se_mamba_pp'


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
        val_pbar = make_progress_bar(
            total=len(validation_loader),
            desc='Validation {}/{}'.format(epoch + 1, total_epochs),
            unit='sample',
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


def train(a, h):
    configure_runtime()
    rank = h.rank
    local_rank = h.local_rank
    world_size = h.num_gpus
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(local_rank)

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

    if world_size > 1:
        device_barrier(device, world_size)
        ddp_kwargs = {
            'device_ids': [local_rank],
            'output_device': local_rank,
            'broadcast_buffers': False,
            'static_graph': True,
        }
        generator = DistributedDataParallel(generator, **ddp_kwargs)
        mssbcqtd = DistributedDataParallel(mssbcqtd, **ddp_kwargs)
        mrd = DistributedDataParallel(mrd, **ddp_kwargs)

    adamw_kwargs = {
        'lr': h.learning_rate,
        'betas': [h.adam_b1, h.adam_b2],
        'fused': True,
    }
    optim_g = torch.optim.AdamW(generator.parameters(), **adamw_kwargs)
    optim_d = torch.optim.AdamW(
        itertools.chain(mrd.parameters(), mssbcqtd.parameters()),
        **adamw_kwargs,
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
    if rank == 0 and a.dataset == 'voicebank':
        logger.info('train clean dir: %s', trainset.clean_wavs_dir)
        logger.info('train noisy dir: %s', trainset.noisy_wavs_dir)
        logger.info('valid clean dir: %s', validset.clean_wavs_dir)
        logger.info('valid noisy dir: %s', validset.noisy_wavs_dir)
    train_sampler = DistributedSampler(trainset) if world_size > 1 else None
    loader_kwargs = {}
    if h.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = getattr(h, 'prefetch_factor', 2)
        loader_kwargs['worker_init_fn'] = worker_init_fn
    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True,
        **loader_kwargs,
    )
    valid_sampler = (
        DistributedSampler(validset, shuffle=False, drop_last=False)
        if world_size > 1 else None
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

        if world_size > 1:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            train_pbar = make_progress_bar(
                total=len(train_loader) * h.batch_size,
                desc='Epoch {}/{}'.format(epoch + 1, a.training_epochs),
                unit='sample',
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

            optim_d.zero_grad(set_to_none=True)
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

            optim_g.zero_grad(set_to_none=True)
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
                train_pbar.update(h.batch_size)
                mag_error = loss_mag.item()
                pha_error = loss_pha.item()
                com_error = loss_com.item() / 2
                con_error = loss_con.item() / 2
                mel_error = mel_loss.item()
                gen_error = loss_gen_all.item()
                disc_error = loss_disc_all.item()
                adv_error = adv_g_loss.item()
                fm_error = fm_g_loss.item()
                postfix = format_postfix(
                    step=steps + 1,
                    gen=gen_error,
                    disc=disc_error,
                    adv=adv_error,
                    fm=fm_error,
                    mag=mag_error,
                    pha=pha_error,
                    com=com_error,
                    mel=mel_error,
                    con=con_error,
                )
                train_pbar.set_postfix(postfix, refresh=True)

                if steps % a.summary_interval == 0:
                    time_error = F.l1_loss(clean_audio, audio_g).item()
                    sw.add_scalar('Training/Generator Loss', gen_error, steps)
                    sw.add_scalar(
                        'Training/Discriminator Loss',
                        disc_error,
                        steps,
                    )
                    sw.add_scalar('Training/adv_g_loss', adv_error, steps)
                    sw.add_scalar('Training/fm_g_loss', fm_error, steps)
                    sw.add_scalar('Training/Magnitude Loss', mag_error, steps)
                    sw.add_scalar('Training/Phase Loss', pha_error, steps)
                    sw.add_scalar('Training/Complex Loss', com_error, steps)
                    sw.add_scalar('Training/Consistency Loss', con_error, steps)
                    sw.add_scalar('Training/Mel Loss', mel_error, steps)
                    sw.add_scalar('Training/Time Loss', time_error, steps)
                    logger.info('%s', train_pbar)

            steps += 1

        if rank == 0:
            train_pbar.close()

        validation_loader = DataLoader(
            validset,
            num_workers=0,
            shuffle=False,
            sampler=valid_sampler,
            batch_size=1,
            pin_memory=True,
            drop_last=False,
        )
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
        del validation_loader

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
        if world_size > 1:
            device_barrier(device, world_size)

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            logger.info(
                'Time taken for epoch %d is %d sec',
                epoch + 1,
                int(time.time() - start),
            )

    if world_size > 1:
        destroy_process_group()


def main():
    configure_runtime()
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
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--num_workers', default=None, type=int)
    parser.add_argument('--dist_timeout_minutes', default=None, type=int)
    a = parser.parse_args()

    if a.dataset == 'librispeech':
        from src.config import SQL_ROOT
        a.sql_root = str(SQL_ROOT)
        if not a.train_splits:
            parser.error('--train_splits is required for --dataset librispeech')
        if not a.validation_splits:
            parser.error('--validation_splits is required for --dataset librispeech')

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for training.')

    rank, local_rank, world_size = resolve_dist_info()
    torch.cuda.set_device(local_rank)

    with open(a.config) as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    if a.batch_size is not None:
        h.batch_size = a.batch_size
    if a.num_workers is not None:
        h.num_workers = a.num_workers
    h.rank = rank
    h.local_rank = local_rank
    h.num_gpus = world_size
    h.dist_timeout_minutes = resolve_dist_timeout_minutes(
        h,
        override_minutes=a.dist_timeout_minutes,
    )
    h.batch_size = max(1, int(h.batch_size // world_size))

    if world_size > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            timeout=timedelta(minutes=h.dist_timeout_minutes),
            device_id=torch.device('cuda', local_rank),
        )

    if rank == 0:
        checkpoint_path = resolve_checkpoint_path(a.checkpoint_path)
        os.makedirs(checkpoint_path, exist_ok=True)
        objects = [checkpoint_path]
    else:
        objects = [None]
    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)
    a.checkpoint_path = objects[0]

    if rank == 0:
        build_env(a.config, 'config.json', a.checkpoint_path)
        logger = setup_logging(a.checkpoint_path, rank=0)
        tqdm.write('Initializing Training Process..')
        tqdm.write('checkpoints directory: {}'.format(a.checkpoint_path))
        tqdm.write('Batch size per GPU: {}'.format(h.batch_size))
        tqdm.write('num_workers: {}'.format(h.num_workers))
        tqdm.write('world_size: {}'.format(world_size))
        logger.info('Initializing Training Process..')
        logger.info('checkpoints directory: %s', a.checkpoint_path)
        logger.info('Batch size per GPU: %d', h.batch_size)
        logger.info('num_workers: %d', h.num_workers)
        logger.info('world_size: %d', world_size)
        if world_size == 1 and torch.cuda.device_count() > 1:
            logger.info(
                'Detected %d GPUs but WORLD_SIZE=1. '
                'Use torchrun --nproc_per_node=%d for multi-GPU.',
                torch.cuda.device_count(),
                torch.cuda.device_count(),
            )

    torch.random.default_generator.manual_seed(h.seed)
    torch.cuda.manual_seed(h.seed)
    train(a, h)


if __name__ == '__main__':
    main()
