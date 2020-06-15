#!/usr/bin/env python

# wujian@2018

import sys
import types

import argparse
import numpy as np

from pathlib import Path
from urllib import request

from libs.cluster import CacgmmTrainer, permu_aligner
from libs.data_handler import SpectrogramReader, ScriptReader, NumpyReader, NumpyWriter
from libs.utils import get_logger, nextpow2
from libs.opts import StftParser, StrToBoolAction

logger = get_logger(__name__)


def run(args):
    stft_kwargs = {
        "frame_len": args.frame_len,
        "frame_hop": args.frame_hop,
        "round_power_of_two": args.round_power_of_two,
        "window": args.window,
        "center": args.center,
        "transpose": False
    }
    np.random.seed(args.seed)
    spectrogram_reader = SpectrogramReader(args.wav_scp, **stft_kwargs)
    MaskReader = {"numpy": NumpyReader, "kaldi": ScriptReader}
    init_mask_reader = MaskReader[args.fmt](
        args.init_mask) if args.init_mask else None

    num_done = 0
    with NumpyWriter(args.dst_dir) as writer:
        dst_dir = Path(args.dst_dir)
        for key, stft in spectrogram_reader:
            if not (dst_dir / f"{key}.npy").exists():
                init_mask = None
                if init_mask_reader and key in init_mask_reader:
                    init_mask = init_mask_reader[key]
                    # K x T x F => K x F x T
                    init_mask = np.transpose(init_mask, (0, 2, 1))
                    logger.info("Using external mask to initialize cacgmm")
                # stft: N x F x T
                trainer = CacgmmTrainer(stft,
                                        args.num_classes,
                                        gamma=init_mask,
                                        cgmm_init=args.cgmm_init,
                                        update_alpha=args.update_alpha)
                try:
                    # EM progress
                    masks = trainer.train(args.num_iters)
                    # K x F x T => K x T x F
                    masks = np.transpose(masks, (0, 2, 1))
                    # align if needed
                    if args.solve_permu:
                        masks = permu_aligner(masks)
                        logger.info(
                            "Permutation alignment done on each frequency")
                    num_done += 1
                    writer.write(key, masks.astype("float32"))
                    logger.info(f"Training utterance {key} ... Done")
                except np.linalg.LinAlgError:
                    logger.warn(f"Training utterance {key} ... Failed")
            else:
                logger.info(f"Training utterance {key} ... Skip")
    logger.info(
        f"Train {num_done:d} utterances over {len(spectrogram_reader):d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Speaker masks estimation using Complex Angular "
        "Central Gaussian Mixture Model (CACGMM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("wav_scp",
                        type=str,
                        help="Multi-channel wave scripts in kaldi format")
    parser.add_argument("dst_dir",
                        type=str,
                        help="Where to dump estimated speech masks")
    parser.add_argument("--num-iters",
                        type=int,
                        default=50,
                        help="Number of iterations to train Cacgmm")
    parser.add_argument("--num-classes",
                        type=int,
                        default=2,
                        help="Number of the cluster "
                        "used in cacgmm model")
    parser.add_argument("--seed",
                        type=int,
                        default=777,
                        help="Random seed for initialization")
    parser.add_argument("--init-mask",
                        type=str,
                        default="",
                        dest="init_mask",
                        help="Mask scripts for cacgmm initialization")
    parser.add_argument("--cgmm-init",
                        action=StrToBoolAction,
                        default=False,
                        help="For 2 classes, using the cgmm init way")
    parser.add_argument("--solve-permu",
                        action=StrToBoolAction,
                        default=True,
                        help="If true, solving permutation problems")
    parser.add_argument("--update-alpha",
                        action=StrToBoolAction,
                        default=True,
                        help="If true, update alpha in M-step")
    parser.add_argument("--mask-format",
                        type=str,
                        dest="fmt",
                        default="numpy",
                        choices=["kaldi", "numpy"],
                        help="Mask storage format")
    args = parser.parse_args()
    run(args)