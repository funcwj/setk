#!/usr/bin/env python

# wujian@2018
"""
Do GWPE Dereverbration Algorithm
"""
import argparse
from distutils.util import strtobool

import numpy as np

from libs.data_handler import SpectrogramReader, WaveWriter
from libs.opts import StftParser
from libs.utils import get_logger, inverse_stft
from libs.wpe import facted_wpd

logger = get_logger(__name__)


def run(args):
    stft_kwargs = {
        "frame_len": args.frame_len,
        "frame_hop": args.frame_hop,
        "window": args.window,
        "center": args.center,  # false to comparable with kaldi
        "transpose": True  # T x F
    }
    spectrogram_reader = SpectrogramReader(
        args.wav_scp,
        round_power_of_two=args.round_power_of_two,
        **stft_kwargs)

    num_done = 0
    with WaveWriter(args.dst_dir, sr=args.sr) as writer:
        for key, obs in spectrogram_reader:
            logger.info(f"Processing utt {key}...")
            if obs.ndim != 3:
                raise RuntimeError(f"Expected 3D array, but got {obs.ndim}")
            try:
                # N x T x F => T x F
                tf_mask, wpd_enh = facted_wpd(obs,
                                              wpd_iters=args.wpd_iters,
                                              cgmm_iters=args.cgmm_iters,
                                              update_alpha=args.update_alpha,
                                              context=args.context,
                                              taps=args.taps,
                                              delay=args.delay)
            except np.linalg.LinAlgError:
                logger.warn(f"{key}: Failed cause LinAlgError in wpd")
                continue
            norm = spectrogram_reader.maxabs(key)
            # dump multi-channel
            samps = inverse_stft(wpd_enh, norm=norm, **stft_kwargs)
            writer.write(key, samps)
            if args.dump_mask:
                np.save(f"{args.dst_dir}/{key}", tf_mask[..., 0])
            # show progress cause slow speed
            num_done += 1
            if not num_done % 100:
                logger.info(f"Processed {num_done:d} utterances...")
    logger.info(
        f"Processed {num_done:d} utterances over {len(spectrogram_reader):d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to do joint dereverbration & denoising algorithm "
                    "(facted form of WPD)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("wav_scp",
                        type=str,
                        help="Multi-channel rspecifier in kaldi format")
    parser.add_argument("dst_dir",
                        type=str,
                        help="Location to dump enhanced audio")
    parser.add_argument("--taps",
                        default=10,
                        type=int,
                        help="Value of taps used in WPE")
    parser.add_argument("--delay",
                        default=3,
                        type=int,
                        help="Value of delay used in WPE")
    parser.add_argument("--context",
                        default=1,
                        type=int,
                        help="Context value to compute PSD "
                             "matrix in WPE algorithm")
    parser.add_argument("--wpd-iters",
                        default=3,
                        type=int,
                        help="Number of iterations for WPD")
    parser.add_argument("--cgmm-iters",
                        default=20,
                        type=int,
                        help="Number of iterations for WPD")
    parser.add_argument("--update-alpha",
                        type=strtobool,
                        default=False,
                        help="If true, update alpha in M-step")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample rate of the input audio")
    parser.add_argument("--dump-mask",
                        default=False,
                        type=strtobool,
                        help="Dump cgmm mask or not")
    args = parser.parse_args()
    run(args)
