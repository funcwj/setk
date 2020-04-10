#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import argparse

import numpy as np

from libs.utils import inverse_stft, get_logger
from libs.opts import StftParser
from libs.data_handler import SpectrogramReader, WaveWriter, ScpReader
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
    # F x N or B x F x N
    weights = np.load(args.weights)
    if weights.ndim == 2:
        beamformer = FixedBeamformer(weights)
        beam_index = None
    else:
        beamformer = [FixedBeamformer(w) for w in weights]
        if not args.beam:
            raise RuntimeError(
                "--beam must be assigned, as there are multiple beams")
        beam_index = ScpReader(args.beam, value_processor=int)
    with WaveWriter(args.dst_dir) as writer:
        for key, stft_mat in spectrogram_reader:
            logger.info(f"Processing utterance {key}...")
            if beamformer:
                beam = beam_index[key]
                stft_enh = beamformer[beam].run(stft_mat)
            else:
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
                        help="Multi-channel wave scripts in Kaldi format")
    parser.add_argument("weights",
                        type=str,
                        help="Fixed beamformer weights in numpy format " +
                        "(in shape F x M or B x F x M)")
    parser.add_argument("dst_dir",
                        type=str,
                        help="Location to dump the enhanced audio")
    parser.add_argument("--beam",
                        type=str,
                        default="",
                        help="Beam index to use in beamformer weights "
                        "(in shape B x F x M)")
    args = parser.parse_args()
    run(args)
