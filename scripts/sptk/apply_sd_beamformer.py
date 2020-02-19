#!/usr/bin/env python
# coding=utf-8
# wujian@2020

import os
import argparse

import numpy as np

from libs.utils import inverse_stft, get_logger
from libs.opts import StftParser, str2tuple
from libs.data_handler import SpectrogramReader, WaveWriter, Reader
from libs.beamformer import LinearSDBeamformer

logger = get_logger(__name__)


def run(args):
    stft_kwargs = {
        "frame_len": args.frame_len,
        "frame_hop": args.frame_hop,
        "window": args.window,
        "center": args.center,
        "transpose": False
    }

    utt2doa = None
    doa = None
    if args.utt2doa:
        utt2doa = Reader(args.utt2doa, value_processor=lambda x: float(x))
        logger.info(f"Use utt2doa {args.utt2doa} for each utterance")
    else:
        doa = args.doa
        if doa < 0:
            doa = 180 + doa
        if doa < 0 or doa > 180:
            raise RuntimeError(f"Invalid doa {doa:.2f} for --doa")
        logger.info(f"Use DoA {doa:.2f} for all utterances")

    spectrogram_reader = SpectrogramReader(
        args.wav_scp,
        round_power_of_two=args.round_power_of_two,
        **stft_kwargs)

    done = 0
    topo = str2tuple(args.linear_topo)
    beamformer = LinearSDBeamformer(topo)
    logger.info(f"Initialize channel LinearSDBeamformer for array: {topo}")

    with WaveWriter(args.dst_dir, fs=args.fs) as writer:
        for key, stft_src in spectrogram_reader:
            if utt2doa:
                if key not in utt2doa:
                    continue
                doa = utt2doa[key]
                if doa < 0:
                    doa = 180 + doa
                if doa < 0 or doa > 180:
                    logger.info(f"Invalid doa {doa:.2f} for utterance {key}")
                    continue
            stft_enh = beamformer.run(doa, stft_src, c=args.speed, sr=args.fs)
            done += 1
            norm = spectrogram_reader.maxabs(key)
            samps = inverse_stft(stft_enh, **stft_kwargs, norm=norm)
            writer.write(key, samps)
    logger.info(f"Processed {done} utterances over {len(spectrogram_reader)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to apply supperdirective beamformer (linear array).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("wav_scp",
                        type=str,
                        help="Rspecifier for multi-channel wave file")
    parser.add_argument("dst_dir",
                        type=str,
                        help="Directory to dump enhanced results")
    parser.add_argument("--fs",
                        type=int,
                        default=16000,
                        help="Sample frequency of input wave")
    parser.add_argument("--speed",
                        type=float,
                        default=340,
                        help="Speed of sound")
    parser.add_argument("--linear-topo",
                        type=str,
                        required=True,
                        help="Topology of linear microphone arrays")
    parser.add_argument("--utt2doa",
                        type=str,
                        default="",
                        help="Given DoA for each utterances, in degrees")
    parser.add_argument("--doa",
                        type=float,
                        default=0,
                        help="DoA for all utterances if "
                        "--utt2doa is not assigned")
    args = parser.parse_args()
    run(args)