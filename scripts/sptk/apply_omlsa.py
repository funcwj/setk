#!/usr/bin/env python

# wujian@2020

import yaml
import argparse

from libs.ns import OMLSA
from libs.opts import StftParser
from libs.utils import inverse_stft, get_logger
from libs.data_handler import SpectrogramReader, NumpyWriter, WaveWriter

logger = get_logger(__name__)


def run(args):
    if args.sr != 16000:
        raise ValueError("Now only support audio in 16kHz")
    # shape: T x F, complex
    stft_kwargs = {
        "frame_len": args.frame_len,
        "frame_hop": args.frame_hop,
        "window": args.window,
        "center": args.center,
    }
    spectrogram_reader = SpectrogramReader(
        args.wav_scp,
        **stft_kwargs,
        round_power_of_two=args.round_power_of_two)

    if args.conf:
        with open(args.conf, "r") as conf:
            omlsa_conf = yaml.full_load(conf)
            suppressor = OMLSA(**omlsa_conf)
    else:
        suppressor = OMLSA()

    if args.output == "wave":
        with WaveWriter(args.dst_dir, fs=args.sr) as writer:
            for key, stft in spectrogram_reader:
                logger.info(f"Processing utterance {key}...")
                gain = suppressor.run(stft)
                samps = inverse_stft(gain * stft, **stft_kwargs)
                writer.write(key, samps)
    else:
        with NumpyWriter(args.dst_dir) as writer:
            for key, stft in spectrogram_reader:
                logger.info(f"Processing utterance {key}...")
                gain = suppressor.run(stft)
                writer.write(key, gain)
    logger.info(f"Processed {len(spectrogram_reader):d} utterances done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to do noise suppression using OMLSA with iMCRA "
        "noise spectrum estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("wav_scp",
                        type=str,
                        help="Noisy audio scripts in Kaldi format")
    parser.add_argument("dst_dir",
                        type=str,
                        help="Location to dump enhanced audio "
                        "or gain coefficients")
    parser.add_argument("--conf",
                        type=str,
                        default="",
                        help="Yaml configurations for OMLSA")
    parser.add_argument("--output",
                        type=str,
                        choices=["gain", "wave"],
                        default="wave",
                        help="Output type of the command")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Waveform data sample frequency")
    args = parser.parse_args()
    run(args)