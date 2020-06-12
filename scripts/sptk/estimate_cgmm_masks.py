#!/usr/bin/env python

# wujian@2018

import argparse
import numpy as np

from pathlib import Path
from libs.cluster import CgmmTrainer
from libs.data_handler import SpectrogramReader, ScriptReader, NumpyReader, NumpyWriter
from libs.utils import get_logger
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
                    if args.trans_mask:
                        init_mask = np.transpose(init_mask)
                    logger.info(
                        "Using external speech mask to initialize cgmm")
                # stft: N x F x T
                trainer = CgmmTrainer(stft, gamma=init_mask)
                try:
                    speech_masks = trainer.train(args.num_iters)
                    num_done += 1
                    speech_masks = np.transpose(speech_masks)
                    writer.write(key, speech_masks.astype(np.float32))
                    logger.info(f"Training utterance {key} ... Done")
                except RuntimeError:
                    logger.warn(f"Training utterance {key} ... Failed")
            else:
                logger.info(f"Training utterance {key} ... Skip")
    logger.info(
        f"Train {num_done:d} utterances over {len(spectrogram_reader):d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Speech & Noise mask estimation using CGMM model "
        "(also see: estimate_cacgmm_masks.py)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("wav_scp",
                        type=str,
                        help="Multi-channel wave scripts in kaldi format")
    parser.add_argument("dst_dir",
                        type=str,
                        help="Location to dump estimated speech masks")
    parser.add_argument("--num-iters",
                        type=int,
                        default=20,
                        help="Number of iterations to train CGMM parameters")
    parser.add_argument("--seed",
                        type=int,
                        default=777,
                        help="Random seed for initialization")
    parser.add_argument("--init-speech-mask",
                        type=str,
                        default="",
                        dest="init_mask",
                        help="Speech mask scripts for cgmm initialization")
    parser.add_argument("--trans-mask",
                        action=StrToBoolAction,
                        default=False,
                        help="If true, transpose initial TF masks")
    parser.add_argument("--mask-format",
                        type=str,
                        dest="fmt",
                        default="numpy",
                        choices=["kaldi", "numpy"],
                        help="Mask storage format")
    args = parser.parse_args()
    run(args)