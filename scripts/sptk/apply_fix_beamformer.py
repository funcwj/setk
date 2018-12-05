#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import argparse
import os

import numpy as np
from scipy.io import loadmat

from libs.utils import istft, get_logger
from libs.opts import get_stft_parser
from libs.data_handler import SpectrogramReader, WaveWriter
from libs.beamformer import FixedBeamformer

logger = get_logger(__name__)


def run(args):
    stft_kwargs = {
        "frame_length": args.frame_length,
        "frame_shift": args.frame_shift,
        "window": args.window,
        "center": args.center,
        "transpose": False
    }
    spectrogram_reader = SpectrogramReader(args.wav_scp, **stft_kwargs)
    weights_dict = loadmat(args.weights)
    if args.weight_key not in weights_dict:
        raise KeyError("Weight key error: no \'{}\' in {}".format(
            args.weight_key, args.weights))

    beamformer = FixedBeamformer(weights_dict[args.weight_key])
    with WaveWriter(args.dump_dir) as writer:
        for key, stft_mat in spectrogram_reader:
            logger.info("Processing utterance {}...".format(key))
            stft_enh = beamformer.run(stft_mat)
            # do not normalize
            samps = istft(stft_enh, **stft_kwargs)
            writer.write(key, samps)
    logger.info("Processed {:d} utterances".format(len(spectrogram_reader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to run fixed beamformer. Runing this command needs "
        "design fixed beamformer first.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[get_stft_parser()])
    parser.add_argument(
        "wav_scp", type=str, help="Multi-channel wave scripts in kaldi format")
    parser.add_argument(
        "weights", type=str, help="Fixed beamformer weight in MATLAB format")
    parser.add_argument(
        "dst_dir", type=str, help="Location to dump enhanced wave file")
    parser.add_argument(
        "--weight-key",
        default="weights",
        help="String key to index matrix in MATLAB's .mat file")
    args = parser.parse_args()
    run(args)
