#!/usr/bin/env python
# coding=utf-8
# gleb-shnshn@2021

import argparse
import math
from distutils.util import strtobool

import numpy as np

from libs.beamformer import LinearDSBeamformer, CircularDSBeamformer, LinearSDBeamformer, CircularSDBeamformer
from libs.data_handler import SpectrogramReader, WaveWriter, ScpReader
from libs.opts import StftParser, str2tuple
from libs.utils import inverse_stft, get_logger, check_doa

logger = get_logger(__name__)
beamformers = ["ds", "sd"]


def do_online_beamform(beamformer, doa, stft_mat, args):
    chunk_size = args.chunk_len
    num_chunks = math.ceil(stft_mat.shape[-1] / chunk_size)
    enh_chunks = []
    for c in range(num_chunks):
        base = chunk_size * c
        chunk = beamformer.run(doa[c], stft_mat[:, :, base:base + chunk_size], c=args.speed, sr=args.sr)
        enh_chunks.append(chunk)
    return np.hstack(enh_chunks)


def process_doa(doa, online):
    if online:
        return list(map(float, doa))
    else:
        return float(doa)


def parse_doa(args, online):
    if args.utt2doa:
        reader = ScpReader(args.utt2doa, value_processor=lambda doa: process_doa(doa, online), num_tokens=-1)
        utt2doa = reader.get
        logger.info(f"Use --utt2doa={args.utt2doa} for each utterance")
    else:
        doa = process_doa(args.doa, online)
        utt2doa = lambda _: doa
        logger.info(f"Use --doa={doa:.2f} for all utterances")
    return utt2doa


def run(args):
    stft_kwargs = {
        "frame_len": args.frame_len,
        "frame_hop": args.frame_hop,
        "window": args.window,
        "center": args.center,
        "transpose": False
    }

    supported_beamformer = {
        "ds": {"linear": LinearDSBeamformer(linear_topo=args.linear_topo),
               "circular": CircularDSBeamformer(radius=args.circular_radius,
                                                num_arounded=args.circular_around,
                                                center=args.circular_center)},
        "sd": {"linear": LinearSDBeamformer(linear_topo=args.linear_topo),
               "circular": CircularSDBeamformer(radius=args.circular_radius,
                                                num_arounded=args.circular_around,
                                                center=args.circular_center)}
    }

    beamformer = supported_beamformer[args.beamformer][args.geometry]
    online = args.chunk_len > 0

    utt2doa = parse_doa(args, online)

    spectrogram_reader = SpectrogramReader(
        args.wav_scp,
        round_power_of_two=args.round_power_of_two,
        **stft_kwargs)

    done = 0
    with WaveWriter(args.dst_dir, sr=args.sr) as writer:
        for key, stft_src in spectrogram_reader:
            doa = utt2doa(key)
            if doa is None:
                logger.info(f"Missing doa for utterance {key}")
                continue
            if not check_doa(args.geometry, doa, online):
                logger.info(f"Invalid doa {doa:.2f} for utterance {key}")
                continue
            if online:
                stft_enh = do_online_beamform(beamformer, doa, stft_src, args)
            else:
                stft_enh = beamformer.run(doa, stft_src, c=args.speed, sr=args.sr)
            norm = spectrogram_reader.maxabs(key) if args.normalize else None
            samps = inverse_stft(stft_enh, **stft_kwargs, norm=norm)
            writer.write(key, samps)
            done += 1
    logger.info(f"Processed {done} utterances over {len(spectrogram_reader)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to apply classic beamformer (linear & circular array).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("wav_scp",
                        type=str,
                        help="Rspecifier for multi-channel wave file")
    parser.add_argument("dst_dir",
                        type=str,
                        help="Directory to dump enhanced results")
    parser.add_argument("--beamformer",
                        type=str,
                        default="ds",
                        choices=beamformers,
                        help="Type of classic beamformer to apply")
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
                        type=str2tuple,
                        default=(),
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
                        type=strtobool,
                        default=False,
                        help="Is there a microphone put in the "
                             "center of the circular array?")
    parser.add_argument("--utt2doa",
                        type=str,
                        default="",
                        help="Given DoA for each utterances, in degrees")
    parser.add_argument("--doa",
                        type=str,
                        default="0",
                        help="DoA for all utterances if "
                             "--utt2doa is not assigned")
    parser.add_argument("--normalize",
                        type=strtobool,
                        default=False,
                        help="Normalize stft after enhancement?")
    parser.add_argument("--chunk-len",
                        type=int,
                        default=-1,
                        help="Number frames per chunk "
                             "(for online setups)")
    args = parser.parse_args()
    run(args)
