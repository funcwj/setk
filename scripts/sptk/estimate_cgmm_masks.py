#!/usr/bin/env python

# wujian@2018

import argparse
import os
import numpy as np

from libs.cluster import CgmmTrainer
from libs.data_handler import SpectrogramReader, ScriptReader, NumpyWriter
from libs.utils import get_logger
from libs.opts import StftParser

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

    spectrogram_reader = SpectrogramReader(args.wav_scp, **stft_kwargs)
    init_mask_reader = ScriptReader(args.init_mask) if args.init_mask else None

    num_done = 0
    with NumpyWriter(args.dst_dir) as writer:
        for key, stft in spectrogram_reader:
            if not os.path.exists(os.path.join(args.dst_dir, f"{key}.npy")):
                init_mask = None
                if init_mask_reader and key in init_mask_reader:
                    init_mask = init_mask_reader[key]
                    logger.info(
                        "Using external speech mask to initialize cgmm")
                # stft: N x F x T
                trainer = CgmmTrainer(stft, Ms=init_mask)
                try:
                    speech_masks = trainer.train(args.num_epochs)
                    num_done += 1
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
        description="Estimate speech & noise masks using CGMM methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("wav_scp",
                        type=str,
                        help="Multi-channel wave scripts in kaldi format")
    parser.add_argument("dst_dir",
                        type=str,
                        help="Location to dump estimated speech masks")
    parser.add_argument("--num-epochs",
                        type=int,
                        default=20,
                        help="Number of epochs to train CGMM parameters")
    parser.add_argument("--init-speech-mask",
                        type=str,
                        default="",
                        dest="init_mask",
                        help="Speech mask scripts for cgmm initialization")
    args = parser.parse_args()
    run(args)