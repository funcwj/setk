#!/usr/bin/env python
# coding=utf-8
# wujian@2020

import argparse

import numpy as np

from libs.utils import inverse_stft, get_logger
from libs.opts import StftParser, str2tuple, StrToBoolAction
from libs.data_handler import SpectrogramReader, WaveWriter, ScpReader
from libs.beamformer import LinearSDBeamformer, CircularSDBeamformer

logger = get_logger(__name__)


def check_doa(geometry, doa):
    """
    Check value of the DoA
    """
    if doa < 0:
        return False
    if geometry == "linear" and doa > 180:
        return False
    if geometry == "circular" and doa >= 360:
        return False
    return True


def run(args):
    stft_kwargs = {
        "frame_len": args.frame_len,
        "frame_hop": args.frame_hop,
        "window": args.window,
        "center": args.center,
        "transpose": False
    }

    if args.geometry == "linear":
        topo = str2tuple(args.linear_topo)
        beamformer = LinearSDBeamformer(topo)
        logger.info(f"Initialize LinearSDBeamformer for array: {topo}")
    else:
        beamformer = CircularSDBeamformer(args.circular_radius,
                                          args.circular_around,
                                          center=args.circular_center)
        logger.info(
            "Initialize CircularSDBeamformer for " +
            f"radius = {args.circular_radius}, center = {args.circular_center}"
        )

    utt2doa = None
    doa = None
    if args.utt2doa:
        utt2doa = ScpReader(args.utt2doa, value_processor=lambda x: float(x))
        logger.info(f"Use --utt2doa={args.utt2doa} for each utterance")
    else:
        doa = args.doa
        if not check_doa(args.geometry, doa):
            logger.info(f"Invalid doa {doa:.2f} for {args.geometry} array")
        logger.info(f"Use --doa={doa:.2f} for all utterances")

    spectrogram_reader = SpectrogramReader(
        args.wav_scp,
        round_power_of_two=args.round_power_of_two,
        **stft_kwargs)

    done = 0
    with WaveWriter(args.dst_dir, sr=args.sr) as writer:
        for key, stft_src in spectrogram_reader:
            if utt2doa:
                if key not in utt2doa:
                    continue
                doa = utt2doa[key]
                if not check_doa(args.geometry, doa):
                    logger.info(f"Invalid DoA {doa:.2f} for utterance {key}")
                    continue
            stft_enh = beamformer.run(doa, stft_src, c=args.speed, sr=args.sr)
            done += 1
            norm = spectrogram_reader.maxabs(key)
            samps = inverse_stft(stft_enh, **stft_kwargs, norm=norm)
            writer.write(key, samps)
    logger.info(f"Processed {done} utterances over {len(spectrogram_reader)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to apply supperdirective beamformer (linear & circular array).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("wav_scp",
                        type=str,
                        help="Rspecifier for multi-channel wave file")
    parser.add_argument("dst_dir",
                        type=str,
                        help="Directory to dump enhanced results")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample rate of the input wave")
    parser.add_argument("--speed",
                        type=float,
                        default=343,
                        help="Speed of sound")
    parser.add_argument("--geometry",
                        type=str,
                        choices=["linear", "circular"],
                        default="linear",
                        help="Geometry of the microphone array")
    parser.add_argument("--linear-topo",
                        type=str,
                        default="",
                        help="Topology of linear microphone arrays")
    parser.add_argument("--circular-around",
                        type=int,
                        default=6,
                        help="Number of the micriphones in circular arrays")
    parser.add_argument("--circular-radius",
                        type=float,
                        default=0.05,
                        help="Radius of circular array")
    parser.add_argument("--circular-center",
                        action=StrToBoolAction,
                        default=False,
                        help="Is there a microphone put in the "
                        "center of the circular array?")
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