#!/usr/bin/env python

# wujian@2018

import argparse
import os
import numpy as np

from libs.cgmm_trainer import CgmmTrainer
from libs.data_handler import SpectrogramReader
from libs.utils import get_logger

logger = get_logger(__name__)


def run(args):
    stft_kwargs = {
        "frame_length": args.frame_length,
        "frame_shift": args.frame_shift,
        "window": args.window,
        "center": args.center,
        "transpose": False
    }

    if not os.path.exists(args.dst_dir):
        os.makedirs(args.dst_dir)

    spectrogram_reader = SpectrogramReader(args.wav_scp, **stft_kwargs)

    num_done = 0
    for key, stft in spectrogram_reader:
        if not os.path.exists(
                os.path.join(args.dst_dir, "{}.npy".format(key))):
            # stft: N x F x T
            trainer = CgmmTrainer(stft)
            try:
                speech_masks = trainer.train(args.num_epochs)
                num_done += 1
                np.save(
                    os.path.join(args.dst_dir, key),
                    speech_masks.astype(np.float32))
                logger.info("Training utterance {} ... Done".format(key))
            except RuntimeError:
                logger.warn("Training utterance {} ... Failed".format(key))
        else:
            logger.info("Training utterance {} ... Skip".format(key))
    logger.info("Train {:d} utterances over {:d}".format(
        num_done, len(spectrogram_reader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate speech & noise masks using CGMM methods")
    parser.add_argument(
        "wav_scp", type=str, help="Multi-channel wave scripts in kaldi format")
    parser.add_argument(
        'dst_dir', type=str, help="Location to dump enhanced wave files")
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        dest="num_epochs",
        help="Number of epochs to train CGMM parameters")
    parser.add_argument(
        "--frame-length",
        type=int,
        default=1024,
        dest="frame_length",
        help="Frame length in number of samples")
    parser.add_argument(
        "--frame-shift",
        type=int,
        default=256,
        dest="frame_shift",
        help="Frame shift in number of samples")
    parser.add_argument(
        "--center",
        action="store_true",
        default=False,
        dest="center",
        help="\'center\' flags in librosa.stft functions")
    parser.add_argument(
        "--window",
        default="hann",
        dest="window",
        help="Type of window function, see scipy.signal.get_window")
    args = parser.parse_args()
    run(args)