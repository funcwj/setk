#!/usr/bin/env python

# wujian@2018
"""
Do GWPE Dereverbration Algorithm
"""

import argparse
import os

from libs.utils import get_logger, istft
from libs.opts import StftParser
from libs.gwpe import wpe
from libs.data_handler import SpectrogramReader, WaveWriter

import numpy as np

logger = get_logger(__name__)


def run(args):
    stft_kwargs = {
        "frame_len": args.frame_len,
        "frame_hop": args.frame_hop,
        "window": args.window,
        "center": args.center,  # false to comparable with kaldi
        "transpose": True  # T x F
    }
    wpe_kwargs = {
        "num_iters": args.num_iters,
        "context": args.context,
        "taps": args.taps,
        "delay": args.delay
    }
    spectrogram_reader = SpectrogramReader(
        args.wav_scp,
        round_power_of_two=args.round_power_of_two,
        **stft_kwargs)

    num_done = 0
    with WaveWriter(args.dst_dir, fs=args.samp_fs) as writer:
        for key, reverbed in spectrogram_reader:
            logger.info(f"Processing utt {key}...")
            # N x T x F => F x N x T
            reverbed = np.transpose(reverbed, (2, 0, 1))
            try:
                # F x N x T
                dereverb = wpe(reverbed, **wpe_kwargs)
            except np.linalg.LinAlgError:
                logger.warn(f"{key}: Failed cause LinAlgError in wpe")
                continue
            # F x N x T => N x T x F
            dereverb = np.transpose(dereverb, (1, 2, 0))
            # dump multi-channel
            samps = np.stack(
                [istft(spectra, **stft_kwargs) for spectra in dereverb])
            writer.write(key, samps)
            # show progress cause slow speed
            num_done += 1
            if not num_done % 100:
                logger.info(f"Processed {num_done:d} utterances...")
    logger.info(
        f"Processed {num_done:d} utterances over {len(spectrogram_reader):d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to do GWPE dereverbration algorithm(recommended "
        "configuration: 512/128/blackman)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("wav_scp",
                        type=str,
                        help="Multi-channel rspecifier in kaldi format")
    parser.add_argument("dst_dir",
                        type=str,
                        help="Location to dump dereverbrated files")
    parser.add_argument("--taps",
                        default=10,
                        type=int,
                        help="Value of taps used in GWPE algorithm")
    parser.add_argument("--delay",
                        default=3,
                        type=int,
                        help="Value of delay used in GWPE algorithm")
    parser.add_argument("--context",
                        default=1,
                        dest="context",
                        type=int,
                        help="Context value to compute PSD "
                        "matrix in GWPE algorithm")
    parser.add_argument("--num-iters",
                        default=3,
                        type=int,
                        help="Number of iterations to step in GWPE")
    parser.add_argument("--sample-frequency",
                        type=int,
                        default=16000,
                        dest="samp_fs",
                        help="Waveform data sample frequency")
    args = parser.parse_args()
    run(args)
