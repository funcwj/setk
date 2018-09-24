#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import os
import argparse
import random
import pprint

import numpy as np

import scipy.io.wavfile as wf

from libs.data_handler import WaveReader
from libs.utils import EPSILON, MAX_INT16, get_logger, write_wav
from tqdm import tqdm

logger = get_logger(__name__)


def check_args(args):
    num_db_spks = len(args.disturb_spks)
    if num_db_spks == 0:
        logger.info("No disturbing speakers, add noise only")
    elif args.max_spk > num_db_spks:
        raise ValueError(
            "Value of --max-spk exceeds number of disturbing speakers: {} vs {}".
            format(args.max_spk, num_db_spks))
    if num_db_spks:
        assert args.min_spk >= 1 and "Value of --min-spk should larger than zero"
    assert args.iters >= 1 and "Value of --iters should larger than zero"


def add_noise(signal, noise, snr, period=False):
    """
        Add noise for target(A very simple version)
    """
    n_nsamps = noise.size
    s_nsamps = signal.size

    if n_nsamps > s_nsamps:
        signal_seg = signal
        noise_shift = random.randint(0, n_nsamps - s_nsamps)
        noise_seg = noise[noise_shift:noise_shift + s_nsamps]
    else:
        noise_seg = np.zeros(s_nsamps)
        if not period:
            signal_shift = random.randint(0, s_nsamps - n_nsamps)
            signal_seg = signal[signal_shift:signal_shift + n_nsamps]
            noise_seg[signal_shift:signal_shift + n_nsamps] = noise
        else:
            signal_seg = signal
            base = 0
            while base + n_nsamps <= s_nsamps:
                noise_seg[base:base + n_nsamps] = noise
                base += n_nsamps
            noise_seg[base:] = noise[:s_nsamps - base]

    noise_seg = noise_seg * (10**(-snr / 10) * np.linalg.norm(signal_seg, 2) /
                             np.maximum(np.linalg.norm(noise_seg, 2), EPSILON))

    return noise_seg


def run(args):
    target_reader = WaveReader(args.target_spk)
    others_reader = [WaveReader(spk_scp) for spk_scp in args.disturb_spks]

    bg_noise_scp, fg_noise_scp = args.bg_noise, args.fg_noise
    bg_noise_reader = WaveReader(bg_noise_scp) if bg_noise_scp else None
    fg_noise_reader = WaveReader(fg_noise_scp) if fg_noise_scp else None

    # for each iteration
    for it in tqdm(range(args.iters)):
        # for each target utts
        for key, target in target_reader:
            noise = np.zeros_like(target)
            # add noise if exists
            for index, noise_reader in enumerate(
                [bg_noise_reader, fg_noise_reader]):
                if noise_reader:
                    # sample noise
                    # randint: [a, b]
                    noise_index = random.randint(0, len(noise_reader) - 1)
                    bg_or_fg_noise = noise_reader[noise_index]
                    # sample snr
                    snr = random.uniform(args.min_snr, args.max_snr)
                    # add noise
                    noise_seg = add_noise(
                        target, bg_or_fg_noise, snr, period=(index == 0))
                    # accumulate noise
                    noise = noise + noise_seg

            if len(others_reader):
                # sample speaker
                num_samp_spk = random.randint(args.min_spk, args.max_spk)
                samp_reader = random.sample(others_reader, num_samp_spk)
                # for each interference speaker
                for spk_noise_reader in samp_reader:
                    # sample interference
                    utt_index = random.randint(0, len(spk_noise_reader) - 1)
                    spk_noise = spk_noise_reader[utt_index]
                    # sample sdr
                    sdr = random.uniform(args.min_sdr, args.max_sdr)
                    # add interference
                    noise_seg = add_noise(target, spk_noise, sdr)
                    # accumulate noise
                    noise = noise + noise_seg
            # sample norm
            sample_norm = random.uniform(0.6, 0.9)
            coef = sample_norm / np.maximum(
                np.linalg.norm(noise, np.inf), np.linalg.norm(target, np.inf))
            write_wav(
                os.path.join(args.target_dump_dir, '{}_{:d}.wav'.format(
                    key, it)), target * coef)
            write_wav(
                os.path.join(args.noise_dump_dir, '{}_{:d}.wav'.format(
                    key, it)), noise * coef)
            mixture = (target + noise) * coef
            mixture = sample_norm * mixture / np.linalg.norm(mixture, np.inf)
            write_wav(
                os.path.join(args.noisy_dump_dir, '{}_{:d}.wav'.format(
                    key, it)), mixture)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "Create mixture of speakers & background/foreground noise for target speaker"
    )
    parser.add_argument(
        "target_spk", type=str, help="Target speaker wave scripts")
    parser.add_argument(
        "--disturb-spks",
        type=str,
        dest="disturb_spks",
        default=[],
        nargs="+",
        help="List of disturbing speaker wave scripts")
    parser.add_argument(
        "--target-dump-dir",
        type=str,
        default=None,
        dest="target_dump_dir",
        help="Location to dump target wave files, using for mask computation")
    parser.add_argument(
        "--noise-dump-dir",
        type=str,
        default=None,
        dest="noise_dump_dir",
        help="Location to dump noise wave files, using for mask computation")
    parser.add_argument(
        "--noisy-dump-dir",
        type=str,
        default=None,
        dest="noisy_dump_dir",
        help="Location to dump mixture wave files")
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Number of iterations to add for same utterance")
    parser.add_argument(
        "--bg-noise",
        type=str,
        dest="bg_noise",
        default=None,
        help="If assigned, add background noise for mixture. "
        "Background noise will repeat itself to fill whole utterance")
    parser.add_argument(
        "--fg-noise",
        type=str,
        dest="fg_noise",
        default=None,
        help="If assigned, add foreground noise for mixture")
    parser.add_argument(
        "--min-snr",
        type=float,
        default=-5,
        dest="min_snr",
        help="Minimum SNR to add background noise")
    parser.add_argument(
        "--max-snr",
        type=float,
        default=5,
        dest="max_snr",
        help="Maximum SNR to add background noise")
    parser.add_argument(
        "--min-sdr",
        type=float,
        default=-5,
        dest="min_sdr",
        help="Minimum SDR to add interference speakers")
    parser.add_argument(
        "--max-sdr",
        type=float,
        default=5,
        dest="max_sdr",
        help="Maximum SDR to add interference speakers")
    parser.add_argument(
        "--min-spk",
        type=int,
        default=1,
        dest="min_spk",
        help="Minimum number of interference speakers to mixed")
    parser.add_argument(
        "--max-spk",
        type=int,
        default=4,
        dest="max_spk",
        help="Maximum number of interference speakers to mixed")
    args = parser.parse_args()
    check_args(args)
    logger.info("Arguments in command:\n{}".format(pprint.pformat(vars(args))))
    run(args)
