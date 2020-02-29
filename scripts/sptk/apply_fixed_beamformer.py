#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import argparse

import numpy as np

from libs.utils import inverse_stft, get_logger
from libs.opts import StftParser
from libs.data_handler import SpectrogramReader, WaveWriter
from libs.beamformer import FixedBeamformer

logger = get_logger(__name__)


def run(args):
    stft_kwargs = {
        "frame_len": args.frame_len,
        "frame_hop": args.frame_hop,
        "window": args.window,
        "center": args.center,
        "transpose": False
    }
    spectrogram_reader = SpectrogramReader(
        args.wav_scp,
        round_power_of_two=args.round_power_of_two,
        **stft_kwargs)
    # F x N
    weights = np.load(args.weights)
    beamformer = FixedBeamformer(weights)
    with WaveWriter(args.dst_dir) as writer:
        for key, stft_mat in spectrogram_reader:
            logger.info(f"Processing utterance {key}...")
            stft_enh = beamformer.run(stft_mat)
            norm = spectrogram_reader.maxabs(key)
            samps = inverse_stft(stft_enh, **stft_kwargs, norm=norm)
            writer.write(key, samps)
    logger.info(f"Processed {len(spectrogram_reader):d} utterances")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to run fixed beamformer. Runing this command needs "
        "to design fixed beamformer first.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("wav_scp",
                        type=str,
                        help="Multi-channel wave scripts in kaldi format")
    parser.add_argument("weights",
                        type=str,
                        help="Fixed beamformer weight in numpy format")
    parser.add_argument("dst_dir",
                        type=str,
                        help="Location to dump enhanced wave file")
    args = parser.parse_args()
    run(args)
