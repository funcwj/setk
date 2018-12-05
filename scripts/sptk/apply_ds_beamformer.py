#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import os
import argparse

import numpy as np

from libs.utils import istft, get_logger
from libs.opts import get_stft_parser
from libs.data_handler import SpectrogramReader, WaveWriter
from libs.beamformer import DSBeamformer

logger = get_logger(__name__)


def run(args):
    stft_kwargs = {
        "frame_length": args.frame_length,
        "frame_shift": args.frame_shift,
        "window": args.window,
        "center": args.center,
        "transpose": False
    }
    topo = list(map(float, args.linear_topo.split(",")))
    doa = args.doa if args.doa > 0 else 180 + args.doa
    if doa < 0 or doa > 180:
        raise RuntimeError("Illegal value for DoA: {:.2f}".format(args.doa))

    spectrogram_reader = SpectrogramReader(args.wav_scp, **stft_kwargs)
    beamformer = DSBeamformer(topo)
    logger.info("Initialize {:d} channel DSBeamformer".format(len(topo)))

    with WaveWriter(args.dst_dir, fs=args.fs) as writer:
        for key, stft_src in spectrogram_reader:
            stft_enh = beamformer.run(
                doa, stft_src, c=args.speed, sample_rate=args.fs)
            power = spectrogram_reader.power(key)
            samps = istft(stft_enh, **stft_kwargs, power=power)
            writer.write(key, samps)
    logger.info("Processed {:d} utterances".format(len(spectrogram_reader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to apply delay and sum beamformer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[get_stft_parser()])
    parser.add_argument(
        "wav_scp", type=str, help="Rspecifier for multi-channel wave file")
    parser.add_argument(
        "dst_dir", type=str, help="Directory to dump enhanced results")
    parser.add_argument(
        "--fs", type=int, default=16000, help="Sample frequency of input wave")
    parser.add_argument(
        "--speed", type=float, default=240, help="Speed of sound")
    parser.add_argument(
        "--linear-topo",
        type=str,
        required=True,
        help="Topology of linear microphone arrays")
    parser.add_argument(
        "--doa",
        type=float,
        default=90,
        help="Given DoA for DS beamformer, in degrees")
    args = parser.parse_args()
    run(args)