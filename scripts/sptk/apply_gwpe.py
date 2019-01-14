#!/usr/bin/env python

# wujian@2018
"""
Do GWPE Dereverbration Algorithm
"""

import argparse
import os

from libs.utils import get_logger, istft, write_wav
from libs.opts import StftParser
from libs.gwpe import wpe
from libs.data_handler import SpectrogramReader

import numpy as np

logger = get_logger(__name__)


def run(args):
    stft_kwargs = {
        "frame_length": args.frame_length,
        "frame_shift": args.frame_shift,
        "round_power_of_two": args.round_power_of_two,
        "window": args.window,
        "center": args.center,  # false to comparable with kaldi
        "transpose": True  # T x F
    }
    wpe_kwargs = {
        "taps": args.taps,
        "delay": args.delay,
        "iters": args.iters,
        "psd_context": args.context
    }
    spectrogram_reader = SpectrogramReader(args.wav_scp, **stft_kwargs)

    if not os.path.exists(args.dst_dir):
        os.makedirs(args.dst_dir)

    for key, reverbed in spectrogram_reader:
        # N x T x F => F x N x T
        reverbed = np.transpose(reverbed, [2, 0, 1])
        # F x N x T
        dereverb = wpe(reverbed, **wpe_kwargs)
        # F x N x T => N x T x F
        dereverb = np.transpose(dereverb, [1, 2, 0])
        # write for each channel
        for chid in range(dereverb.shape[0]):
            samps = istft(dereverb[chid], **stft_kwargs)
            write_wav(
                os.path.join(args.dst_dir, "{}.CH{:d}.wav".format(
                    key, chid + 1)),
                samps,
                fs=args.samp_freq)
    logger.info("Processed {:d} utterances".format(len(spectrogram_reader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to do GWPE dereverbration algorithm(512/128/blackman)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument(
        "wav_scp", type=str, help="Multi-channel wave scripts in kaldi format")
    parser.add_argument(
        "dst_dir",
        type=str,
        help="Location to dump files after dereverbration")
    parser.add_argument(
        "--taps",
        default=10,
        type=int,
        help="Value of taps used in GWPE algorithm")
    parser.add_argument(
        "--delay",
        default=3,
        type=int,
        help="Value of delay used in GWPE algorithm")
    parser.add_argument(
        "--psd-context",
        default=3,
        dest="context",
        type=int,
        help="Context value to compute PSD matrix in GWPE algorithm")
    parser.add_argument(
        "--iters",
        default=3,
        type=int,
        help="Number of iterations to step in GWPE")
    parser.add_argument(
        "--sample-frequency",
        type=int,
        default=16000,
        dest="samp_freq",
        help="Waveform data sample frequency")
    args = parser.parse_args()
    run(args)
