#!/usr/bin/env python

# wujian@2019
"""
Compute labels for DC (Deep Clustering) training:
-1      means silence
0...N   for each speaker
"""

import argparse

import numpy as np

from libs.opts import StftParser
from libs.data_handler import SpectrogramReader, NumpyWriter
from libs.utils import get_logger, EPSILON

logger = get_logger(__name__)


def run(args):
    # shape: T x F
    stft_kwargs = {
        "frame_len": args.frame_len,
        "frame_hop": args.frame_hop,
        "round_power_of_two": args.round_power_of_two,
        "window": args.window,
        "center": args.center,
        "apply_abs": True,
    }
    spk_scps = args.spks.split(",")
    if len(spk_scps) < 2:
        raise RuntimeError("Please give at least 2 speakers")
    mix_reader = SpectrogramReader(args.mix, **stft_kwargs)
    spk_reader = [SpectrogramReader(spk, **stft_kwargs) for spk in spk_scps]

    with NumpyWriter(args.dir) as writer:
        for key, mix in mix_reader:
            T, F = mix.shape
            masks = np.zeros_like(mix, dtype=np.float32)
            # sil: -1
            mix_2db = 20 * np.log10(np.maximum(mix, EPSILON))
            sil_idx = mix_2db < (np.max(mix_2db) - args.beta)
            masks[sil_idx] = -1
            logger.info("For {}, silence covered {:.2f}%".format(
                key,
                np.sum(sil_idx) * 100 / (T * F)))
            # for each speaker
            act_idx = ~sil_idx
            labels = np.argmax(np.stack([reader[key]
                                         for reader in spk_reader]),
                               axis=0)
            masks[act_idx] = labels[act_idx]
            writer.write(key, masks)
    logger.info("Processed {:d} utterances done".format(len(mix_reader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to compute labels for DC (Deep Clustering) "
        "training, -1 means silence, 0..N for each speaker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("mix", type=str, help="Rspecifier for mixture")
    parser.add_argument("spks",
                        type=str,
                        help="Rspecifier for multiple speakers, "
                        "separated by \',\', egs: spk1.scp,spk2.scp")
    parser.add_argument("dir",
                        type=str,
                        help="Directory to store computed labels")
    parser.add_argument("--beta",
                        type=float,
                        default=40,
                        help="Threshold to discriminate silence bins (in dB)")
    args = parser.parse_args()
    run(args)