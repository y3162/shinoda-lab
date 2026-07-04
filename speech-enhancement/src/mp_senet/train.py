import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import time
import argparse
import json
import logging
import random
import shutil
from datetime import datetime

import librosa
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import numpy as np
from pesq import pesq

from src.mp_senet.model.model import MPNet, phase_losses
from src.mp_senet.model.discriminator import MetricDiscriminator
from src.mp_senet.utils import (
    AttrDict,
    scan_checkpoint,
    load_checkpoint,
    save_checkpoint,
)

torch.backends.cudnn.benchmark = True

CHECKPOINT_ROOT = 'data/checkpoints/mp_senet'


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)


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

    stream_handler = TqdmLoggingHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def resolve_checkpoint_path(checkpoint_root):
    if os.path.isdir(checkpoint_root):
        cp_g = scan_checkpoint(checkpoint_root, 'g_')
        cp_do = scan_checkpoint(checkpoint_root, 'do_')
        if cp_g is not None and cp_do is not None:
            return checkpoint_root

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(checkpoint_root, timestamp)


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


def pesq_score(utts_r, utts_g, h):
    scores = [
        cal_pesq(
            utts_r[i].squeeze().cpu().numpy(),
            utts_g[i].squeeze().cpu().numpy(),
            h.sampling_rate,
        )
        for i in range(len(utts_r))
    ]
    return np.mean(scores)


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
        n_cache_reuse=1,
        device=None,
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
        self.cached_clean_wav = None
        self.cached_noisy_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device

    def __getitem__(self, index):
        filename = self.audio_indexes[index]
        if self._cache_ref_count == 0:
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
            self.cached_clean_wav = clean_audio
            self.cached_noisy_wav = noisy_audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            clean_audio = self.cached_clean_wav
            noisy_audio = self.cached_noisy_wav
            self._cache_ref_count -= 1

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
            init_method=h.dist_config['dist_url'],
            world_size=h.dist_config['world_size'] * h.num_gpus,
            rank=rank,
        )

    logger = setup_logging(a.checkpoint_path, rank=rank)

    torch.cuda.manual_seed(h.seed)
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
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        discriminator.load_state_dict(state_dict_do['discriminator'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

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
        n_cache_reuse=0,
        shuffle=False if h.num_gpus > 1 else True,
        device=device,
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
            n_cache_reuse=0,
            device=device,
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

    epoch_range = range(max(0, last_epoch), a.training_epochs)
    if rank == 0:
        epoch_range = tqdm(
            epoch_range,
            desc='Epochs',
            unit='epoch',
            position=0,
        )

    for epoch in epoch_range:
        if rank == 0:
            start = time.time()
            logger.info('Epoch: %d', epoch + 1)

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        train_iter = train_loader
        if rank == 0:
            train_iter = tqdm(
                train_loader,
                desc='Epoch {}/{}'.format(epoch + 1, a.training_epochs),
                unit='step',
                leave=False,
                position=1,
            )

        for i, batch in enumerate(train_iter):
            if rank == 0:
                start_b = time.time()

            clean_audio, noisy_audio = batch
            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
            noisy_audio = torch.autograd.Variable(noisy_audio.to(device, non_blocking=True))
            one_labels = torch.ones(h.batch_size).to(device, non_blocking=True)

            clean_mag, clean_pha, clean_com = mag_pha_stft(
                clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor,
            )
            noisy_mag, noisy_pha, noisy_com = mag_pha_stft(
                noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor,
            )

            mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)

            audio_g = mag_pha_istft(
                mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor,
            )
            mag_g_hat, pha_g_hat, com_g_hat = mag_pha_stft(
                audio_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor,
            )

            audio_list_r, audio_list_g = (
                list(clean_audio.cpu().numpy()),
                list(audio_g.detach().cpu().numpy()),
            )
            batch_pesq_score = batch_pesq(audio_list_r, audio_list_g)

            optim_d.zero_grad()
            metric_r = discriminator(clean_mag, clean_mag)
            metric_g = discriminator(clean_mag, mag_g_hat.detach())
            loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())

            if batch_pesq_score is not None:
                loss_disc_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten())
            else:
                logger.warning('pesq is None!')
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

            if rank == 0:
                train_iter.set_postfix(
                    step=steps,
                    gen=float(loss_gen_all),
                    disc=float(loss_disc_all),
                    refresh=False,
                )

                need_metrics = (
                    steps % a.stdout_interval == 0
                    or steps % a.summary_interval == 0
                )
                if need_metrics:
                    with torch.no_grad():
                        metric_error = F.mse_loss(metric_g.flatten(), one_labels).item()
                        mag_error = F.mse_loss(clean_mag, mag_g).item()
                        ip_error, gd_error, iaf_error = phase_losses(clean_pha, pha_g)
                        pha_error = (ip_error + gd_error + iaf_error).item()
                        com_error = F.mse_loss(clean_com, com_g).item()
                        time_error = F.l1_loss(clean_audio, audio_g).item()
                        stft_error = F.mse_loss(com_g, com_g_hat).item()

                if steps % a.stdout_interval == 0:
                    logger.info(
                        'Steps: %d, Gen Loss: %.3f, Disc Loss: %.3f, '
                        'Metric loss: %.3f, Magnitude Loss: %.3f, '
                        'Phase Loss: %.3f, Complex Loss: %.3f, '
                        'Time Loss: %.3f, STFT Loss: %.3f, s/b: %.3f',
                        steps,
                        float(loss_gen_all),
                        float(loss_disc_all),
                        metric_error,
                        mag_error,
                        pha_error,
                        com_error,
                        time_error,
                        stft_error,
                        time.time() - start_b,
                    )

                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(
                        checkpoint_path,
                        {
                            'generator': (
                                generator.module if h.num_gpus > 1 else generator
                            ).state_dict()
                        },
                    )
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(
                        checkpoint_path,
                        {
                            'discriminator': (
                                discriminator.module if h.num_gpus > 1 else discriminator
                            ).state_dict(),
                            'optim_g': optim_g.state_dict(),
                            'optim_d': optim_d.state_dict(),
                            'steps': steps,
                            'epoch': epoch,
                        },
                    )
                    logger.info('Saved checkpoint at step %d', steps)

                if steps % a.summary_interval == 0:
                    sw.add_scalar("Training/Generator Loss", loss_gen_all, steps)
                    sw.add_scalar("Training/Discriminator Loss", loss_disc_all, steps)
                    sw.add_scalar("Training/Metric Loss", metric_error, steps)
                    sw.add_scalar("Training/Magnitude Loss", mag_error, steps)
                    sw.add_scalar("Training/Phase Loss", pha_error, steps)
                    sw.add_scalar("Training/Complex Loss", com_error, steps)
                    sw.add_scalar("Training/Time Loss", time_error, steps)
                    sw.add_scalar("Training/Consistency Loss", stft_error, steps)

                if steps % a.validation_interval == 0 and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    audios_r, audios_g = [], []
                    val_mag_err_tot = 0
                    val_pha_err_tot = 0
                    val_com_err_tot = 0
                    val_stft_err_tot = 0
                    with torch.no_grad():
                        val_iter = tqdm(
                            validation_loader,
                            desc='Validation',
                            unit='utt',
                            leave=False,
                            position=2,
                        )
                        for j, batch in enumerate(val_iter):
                            clean_audio, noisy_audio = batch
                            clean_audio = torch.autograd.Variable(
                                clean_audio.to(device, non_blocking=True)
                            )
                            noisy_audio = torch.autograd.Variable(
                                noisy_audio.to(device, non_blocking=True)
                            )

                            clean_mag, clean_pha, clean_com = mag_pha_stft(
                                clean_audio,
                                h.n_fft,
                                h.hop_size,
                                h.win_size,
                                h.compress_factor,
                            )
                            noisy_mag, noisy_pha, noisy_com = mag_pha_stft(
                                noisy_audio,
                                h.n_fft,
                                h.hop_size,
                                h.win_size,
                                h.compress_factor,
                            )

                            mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)

                            audio_g = mag_pha_istft(
                                mag_g,
                                pha_g,
                                h.n_fft,
                                h.hop_size,
                                h.win_size,
                                h.compress_factor,
                            )
                            mag_g_hat, pha_g_hat, com_g_hat = mag_pha_stft(
                                audio_g,
                                h.n_fft,
                                h.hop_size,
                                h.win_size,
                                h.compress_factor,
                            )
                            audios_r += torch.split(clean_audio, 1, dim=0)
                            audios_g += torch.split(audio_g, 1, dim=0)

                            val_mag_err_tot += F.mse_loss(clean_mag, mag_g).item()
                            val_ip_err, val_gd_err, val_iaf_err = phase_losses(
                                clean_pha, pha_g
                            )
                            val_pha_err_tot += (
                                val_ip_err + val_gd_err + val_iaf_err
                            ).item()
                            val_com_err_tot += F.mse_loss(clean_com, com_g).item()
                            val_stft_err_tot += F.mse_loss(com_g, com_g_hat).item()

                        val_mag_err = val_mag_err_tot / (j + 1)
                        val_pha_err = val_pha_err_tot / (j + 1)
                        val_com_err = val_com_err_tot / (j + 1)
                        val_stft_err = val_stft_err_tot / (j + 1)
                        val_pesq_score = pesq_score(audios_r, audios_g, h).item()
                        logger.info(
                            'Steps: %d, PESQ Score: %.3f, s/b: %.3f',
                            steps,
                            val_pesq_score,
                            time.time() - start_b,
                        )
                        sw.add_scalar("Validation/PESQ Score", val_pesq_score, steps)
                        sw.add_scalar("Validation/Magnitude Loss", val_mag_err, steps)
                        sw.add_scalar("Validation/Phase Loss", val_pha_err, steps)
                        sw.add_scalar("Validation/Complex Loss", val_com_err, steps)
                        sw.add_scalar(
                            "Validation/Consistency Loss",
                            val_stft_err,
                            steps,
                        )

                    if epoch >= a.best_checkpoint_start_epoch:
                        if val_pesq_score > best_pesq:
                            best_pesq = val_pesq_score
                            best_checkpoint_path = "{}/g_best".format(a.checkpoint_path)
                            save_checkpoint(
                                best_checkpoint_path,
                                {
                                    'generator': (
                                        generator.module
                                        if h.num_gpus > 1
                                        else generator
                                    ).state_dict()
                                },
                            )
                            logger.info(
                                'Updated best checkpoint (PESQ=%.3f) at step %d',
                                best_pesq,
                                steps,
                            )

                    generator.train()

            steps += 1

            if a.max_steps is not None and steps >= a.max_steps:
                if rank == 0:
                    logger.info(
                        'Reached max_steps=%d. Stopping training.',
                        a.max_steps,
                    )
                break

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            logger.info(
                'Time taken for epoch %d is %d sec',
                epoch + 1,
                int(time.time() - start),
            )

        if a.max_steps is not None and steps >= a.max_steps:
            break

    if h.num_gpus > 1:
        destroy_process_group()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
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
    parser.add_argument('--config', default='src/mp_senet/model/config.json')
    parser.add_argument('--training_epochs', default=400, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument('--best_checkpoint_start_epoch', default=40, type=int)
    parser.add_argument(
        '--max_steps',
        default=None,
        type=int,
        help='Stop training after this many steps (for smoke tests).',
    )
    parser.add_argument(
        '--num_workers',
        default=None,
        type=int,
        help='Override config num_workers for DataLoader.',
    )

    a = parser.parse_args()
    a.checkpoint_path = resolve_checkpoint_path(a.checkpoint_path)
    os.makedirs(a.checkpoint_path, exist_ok=True)

    logger = setup_logging(a.checkpoint_path, rank=0)
    logger.info('Initializing Training Process..')
    logger.info('checkpoints directory: %s', a.checkpoint_path)

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    if a.num_workers is not None:
        h.num_workers = a.num_workers
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        logger.info('Batch size per GPU: %d', h.batch_size)
        logger.info('num_workers: %d', h.num_workers)
    else:
        raise RuntimeError('CUDA is required for training.')

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
