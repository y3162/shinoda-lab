import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import time
import argparse
import json
import logging
import random
import shutil
from datetime import datetime, timedelta

import librosa
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from src.mp_senet.model.model import MPNet, phase_losses
from src.mp_senet.model.discriminator import MetricDiscriminator
from src.mp_senet.utils import (
    AttrDict,
    load_checkpoint,
    save_checkpoint,
    pesq_score,
)

torch.backends.cudnn.benchmark = True

CHECKPOINT_ROOT = 'data/checkpoints/mp_senet'
DEFAULT_DIST_TIMEOUT_MINUTES = 120


def configure_distributed_runtime():
    os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
    os.environ.setdefault('NCCL_P2P_DISABLE', '1')


def dist_init_method(checkpoint_path):
    init_file = os.path.abspath(os.path.join(checkpoint_path, '.dist_init'))
    return 'file://' + init_file


def resolve_dist_timeout_minutes(h, override_minutes=None):
    if override_minutes is not None:
        return override_minutes
    return h.dist_config.get('dist_timeout_minutes', DEFAULT_DIST_TIMEOUT_MINUTES)


def setup_logging(log_dir, rank=0):
    logger = logging.getLogger('mp_senet.train')
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


def save_latest_checkpoint(checkpoint_path, generator, discriminator, optim_g, optim_d, steps, epoch, num_gpus):
    gen = generator.module if num_gpus > 1 else generator
    disc = discriminator.module if num_gpus > 1 else discriminator
    save_checkpoint(
        os.path.join(checkpoint_path, 'g_latest'),
        {'generator': gen.state_dict()},
    )
    save_checkpoint(
        os.path.join(checkpoint_path, 'do_latest'),
        {
            'discriminator': disc.state_dict(),
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
    mag = torch.sqrt(stft_spec.pow(2).sum(-1) + (1e-9))
    pha = torch.atan2(
        stft_spec[:, :, :, 1] + (1e-10),
        stft_spec[:, :, :, 0] + (1e-5),
    )
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    return mag, pha, com


def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    mag = torch.pow(mag, (1.0 / compress_factor))
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(
        com,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
    )
    return wav


def forward_generator_batch(generator, clean_audio, noisy_audio, h):
    clean_mag, clean_pha, clean_com = mag_pha_stft(
        clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor,
    )
    noisy_mag, noisy_pha, _ = mag_pha_stft(
        noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor,
    )

    mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)

    audio_g = mag_pha_istft(
        mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor,
    )
    mag_g_hat, _, com_g_hat = mag_pha_stft(
        audio_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor,
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
        'mag_g_hat': mag_g_hat,
        'com_g_hat': com_g_hat,
    }


def compute_reconstruction_errors(
    clean_mag,
    clean_pha,
    clean_com,
    mag_g,
    pha_g,
    com_g,
    com_g_hat,
    clean_audio=None,
    audio_g=None,
):
    mag_error = F.mse_loss(clean_mag, mag_g).item()
    ip_error, gd_error, iaf_error = phase_losses(clean_pha, pha_g)
    pha_error = (ip_error + gd_error + iaf_error).item()
    com_error = F.mse_loss(clean_com, com_g).item()
    stft_error = F.mse_loss(com_g, com_g_hat).item()

    errors = {
        'mag_error': mag_error,
        'pha_error': pha_error,
        'com_error': com_error,
        'stft_error': stft_error,
    }
    if clean_audio is not None and audio_g is not None:
        errors['time_error'] = F.l1_loss(clean_audio, audio_g).item()
    return errors


def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_indexes = [
            x.split('|')[0].split()[0]
            for x in fi.read().split('\n')
            if len(x) > 0
        ]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_indexes = [
            x.split('|')[0].split()[0]
            for x in fi.read().split('\n')
            if len(x) > 0
        ]

    return training_indexes, validation_indexes


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        training_indexes,
        clean_wavs_dir,
        noisy_wavs_dir,
        segment_size,
        sampling_rate,
        split=True,
        shuffle=True,
    ):
        self.audio_indexes = training_indexes
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_indexes)
        self.clean_wavs_dir = clean_wavs_dir
        self.noisy_wavs_dir = noisy_wavs_dir
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split

    def __getitem__(self, index):
        filename = self.audio_indexes[index]
        clean_audio, _ = librosa.load(
            os.path.join(self.clean_wavs_dir, filename + '.wav'),
            sr=self.sampling_rate,
        )
        noisy_audio, _ = librosa.load(
            os.path.join(self.noisy_wavs_dir, filename + '.wav'),
            sr=self.sampling_rate,
        )
        length = min(len(clean_audio), len(noisy_audio))
        clean_audio, noisy_audio = clean_audio[:length], noisy_audio[:length]

        clean_audio, noisy_audio = (
            torch.FloatTensor(clean_audio),
            torch.FloatTensor(noisy_audio),
        )
        norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
        clean_audio = (clean_audio * norm_factor).unsqueeze(0)
        noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)

        assert clean_audio.size(1) == noisy_audio.size(1)

        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                clean_audio = clean_audio[:, audio_start: audio_start + self.segment_size]
                noisy_audio = noisy_audio[:, audio_start: audio_start + self.segment_size]
            else:
                clean_audio = torch.nn.functional.pad(
                    clean_audio,
                    (0, self.segment_size - clean_audio.size(1)),
                    'constant',
                )
                noisy_audio = torch.nn.functional.pad(
                    noisy_audio,
                    (0, self.segment_size - noisy_audio.size(1)),
                    'constant',
                )

        return (clean_audio.squeeze(), noisy_audio.squeeze())

    def __len__(self):
        return len(self.audio_indexes)


def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_init_method,
            world_size=h.dist_config['world_size'] * h.num_gpus,
            rank=rank,
            timeout=timedelta(minutes=h.dist_timeout_minutes),
        )

    logger = setup_logging(a.checkpoint_path, rank=rank)

    torch.cuda.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = MPNet(h).to(device)
    discriminator = MetricDiscriminator().to(device)

    if rank == 0:
        logger.info(generator)
        num_params = 0
        for p in generator.parameters():
            num_params += p.numel()
        logger.info('Total Parameters: {:.3f}M'.format(num_params / 1e6))
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
        discriminator.load_state_dict(state_dict_do['discriminator'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
        if rank == 0:
            logger.info(
                'Resuming from step %d, epoch %d',
                state_dict_do['steps'],
                state_dict_do['epoch'] + 1,
            )

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(
        generator.parameters(),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )
    optim_d = torch.optim.AdamW(
        discriminator.parameters(),
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

    training_indexes, validation_indexes = get_dataset_filelist(a)

    trainset = Dataset(
        training_indexes,
        a.input_clean_wavs_dir,
        a.input_noisy_wavs_dir,
        h.segment_size,
        h.sampling_rate,
        split=True,
        shuffle=False if h.num_gpus > 1 else True,
    )

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=False,
        sampler=train_sampler,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True,
    )

    if rank == 0:
        validset = Dataset(
            validation_indexes,
            a.input_validation_clean_wavs_dir,
            a.input_validation_noisy_wavs_dir,
            h.segment_size,
            h.sampling_rate,
            split=False,
            shuffle=False,
        )

        validation_loader = DataLoader(
            validset,
            num_workers=1,
            shuffle=False,
            sampler=None,
            batch_size=1,
            pin_memory=True,
            drop_last=True,
        )

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    discriminator.train()

    best_pesq = 0
    epoch = last_epoch

    for epoch in range(max(0, last_epoch), a.training_epochs):
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
            one_labels = torch.ones(h.batch_size).to(device, non_blocking=True)

            outputs = forward_generator_batch(generator, clean_audio, noisy_audio, h)
            clean_mag = outputs['clean_mag']
            clean_pha = outputs['clean_pha']
            clean_com = outputs['clean_com']
            mag_g = outputs['mag_g']
            pha_g = outputs['pha_g']
            com_g = outputs['com_g']
            audio_g = outputs['audio_g']
            mag_g_hat = outputs['mag_g_hat']
            com_g_hat = outputs['com_g_hat']

            optim_d.zero_grad()
            metric_r = discriminator(clean_mag, clean_mag)
            metric_g = discriminator(clean_mag, mag_g_hat.detach())
            loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())
            # Step-wise PESQ calculation is intentionally disabled to avoid
            # CPU-side blocking and GPU idle time.
            loss_disc_g = 0

            loss_disc_all = loss_disc_r + loss_disc_g
            loss_disc_all.backward()
            optim_d.step()

            optim_g.zero_grad()

            loss_mag = F.mse_loss(clean_mag, mag_g)
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g)
            loss_pha = loss_ip + loss_gd + loss_iaf
            loss_com = F.mse_loss(clean_com, com_g) * 2
            loss_stft = F.mse_loss(com_g, com_g_hat) * 2
            loss_time = F.l1_loss(clean_audio, audio_g)
            metric_g = discriminator(clean_mag, mag_g_hat)
            loss_metric = F.mse_loss(metric_g.flatten(), one_labels)

            loss_gen_all = (
                loss_mag * 0.9
                + loss_pha * 0.3
                + loss_com * 0.1
                + loss_stft * 0.1
                + loss_metric * 0.05
                + loss_time * 0.2
            )

            loss_gen_all.backward()
            optim_g.step()

            if h.num_gpus > 1:
                barrier()

            if rank == 0:
                with torch.no_grad():
                    metric_error = F.mse_loss(metric_g.flatten(), one_labels).item()
                    errors = compute_reconstruction_errors(
                        clean_mag,
                        clean_pha,
                        clean_com,
                        mag_g,
                        pha_g,
                        com_g,
                        com_g_hat,
                        clean_audio=clean_audio,
                        audio_g=audio_g,
                    )
                    mag_error = errors['mag_error']
                    pha_error = errors['pha_error']
                    com_error = errors['com_error']
                    stft_error = errors['stft_error']
                    time_error = errors['time_error']

                if steps % a.summary_interval == 0:
                    sw.add_scalar("Training/Generator Loss", loss_gen_all, steps)
                    sw.add_scalar("Training/Discriminator Loss", loss_disc_all, steps)
                    sw.add_scalar("Training/Metric Loss", metric_error, steps)
                    sw.add_scalar("Training/Magnitude Loss", mag_error, steps)
                    sw.add_scalar("Training/Phase Loss", pha_error, steps)
                    sw.add_scalar("Training/Complex Loss", com_error, steps)
                    sw.add_scalar("Training/Time Loss", time_error, steps)
                    sw.add_scalar("Training/Consistency Loss", stft_error, steps)

            if h.num_gpus > 1:
                barrier()

            steps += 1

            if rank == 0:
                train_pbar.set_postfix(
                    format_postfix(
                        step=steps,
                        gen=float(loss_gen_all),
                        disc=float(loss_disc_all),
                        metric=metric_error,
                        mag=mag_error,
                        pha=pha_error,
                        com=com_error,
                        time=time_error,
                        stft=stft_error,
                    ),
                    refresh=False,
                )
                train_pbar.update(h.batch_size)

        if rank == 0:
            train_pbar.close()

        if rank == 0:
            generator.eval()
            torch.cuda.empty_cache()
            audios_r, audios_g = [], []
            val_mag_err_tot = 0
            val_pha_err_tot = 0
            val_com_err_tot = 0
            val_stft_err_tot = 0
            with torch.no_grad():
                for j, batch in enumerate(validation_loader):
                    clean_audio, noisy_audio = batch
                    clean_audio = clean_audio.to(device, non_blocking=True)
                    noisy_audio = noisy_audio.to(device, non_blocking=True)

                    outputs = forward_generator_batch(
                        generator, clean_audio, noisy_audio, h,
                    )
                    errors = compute_reconstruction_errors(
                        outputs['clean_mag'],
                        outputs['clean_pha'],
                        outputs['clean_com'],
                        outputs['mag_g'],
                        outputs['pha_g'],
                        outputs['com_g'],
                        outputs['com_g_hat'],
                    )
                    audios_r += torch.split(clean_audio, 1, dim=0)
                    audios_g += torch.split(outputs['audio_g'], 1, dim=0)

                    val_mag_err_tot += errors['mag_error']
                    val_pha_err_tot += errors['pha_error']
                    val_com_err_tot += errors['com_error']
                    val_stft_err_tot += errors['stft_error']

            val_mag_err = val_mag_err_tot / (j + 1)
            val_pha_err = val_pha_err_tot / (j + 1)
            val_com_err = val_com_err_tot / (j + 1)
            val_stft_err = val_stft_err_tot / (j + 1)
            val_pesq_score = pesq_score(audios_r, audios_g, h).item()

            val_message = (
                'Validation (epoch {}/{}): PESQ={:.3f}, mag={:.3f}, '
                'pha={:.3f}, com={:.3f}, stft={:.3f}'
            ).format(
                epoch + 1,
                a.training_epochs,
                val_pesq_score,
                val_mag_err,
                val_pha_err,
                val_com_err,
                val_stft_err,
            )
            tqdm.write(val_message)
            logger.info(val_message)

            sw.add_scalar("Validation/PESQ Score", val_pesq_score, epoch + 1)
            sw.add_scalar("Validation/Magnitude Loss", val_mag_err, epoch + 1)
            sw.add_scalar("Validation/Phase Loss", val_pha_err, epoch + 1)
            sw.add_scalar("Validation/Complex Loss", val_com_err, epoch + 1)
            sw.add_scalar("Validation/Consistency Loss", val_stft_err, epoch + 1)

            if val_pesq_score > best_pesq:
                best_pesq = val_pesq_score
                save_best_checkpoint(
                    a.checkpoint_path,
                    generator,
                    h.num_gpus,
                )
                best_message = (
                    'Updated best checkpoint (PESQ={:.3f}) at epoch {}'
                ).format(best_pesq, epoch + 1)
                tqdm.write(best_message)
                logger.info(best_message)

            save_latest_checkpoint(
                a.checkpoint_path,
                generator,
                discriminator,
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
            barrier()

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            epoch_time = int(time.time() - start)
            logger.info(
                'Time taken for epoch %d is %d sec',
                epoch + 1,
                epoch_time,
            )

    if h.num_gpus > 1:
        barrier()
        destroy_process_group()


def main():
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--checkpoint_path', default=CHECKPOINT_ROOT)
    parser.add_argument('--config', default='src/mp_senet/config_conformer.json')
    parser.add_argument('--training_epochs', default=400, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument(
        '--num_workers',
        default=None,
        type=int,
        help='Override config num_workers for DataLoader.',
    )
    parser.add_argument(
        '--dist_timeout_minutes',
        default=None,
        type=int,
        help=(
            'NCCL/process-group timeout in minutes for barrier and collective ops. '
            'Increase when running multiple jobs on one host.'
        ),
    )

    a = parser.parse_args()
    a.checkpoint_path = resolve_checkpoint_path(a.checkpoint_path)
    os.makedirs(a.checkpoint_path, exist_ok=True)

    logger = setup_logging(a.checkpoint_path, rank=0)
    tqdm.write('Initializing Training Process..')
    tqdm.write('checkpoints directory: {}'.format(a.checkpoint_path))
    logger.info('Initializing Training Process..')
    logger.info('checkpoints directory: %s', a.checkpoint_path)

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        tqdm.write('Batch size per GPU: {}'.format(h.batch_size))
        tqdm.write('num_workers: {}'.format(h.num_workers))
        logger.info('Batch size per GPU: %d', h.batch_size)
        logger.info('num_workers: %d', h.num_workers)
    else:
        raise RuntimeError('CUDA is required for training.')

    if h.num_gpus > 1:
        configure_distributed_runtime()
        tqdm.write('dist_init_method: {}'.format(h.dist_init_method))
        tqdm.write('dist_timeout_minutes: {}'.format(h.dist_timeout_minutes))
        logger.info('dist_init_method: %s', h.dist_init_method)
        logger.info('dist_timeout_minutes: %d', h.dist_timeout_minutes)
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
