#!/usr/bin/env python

# wujian@2020

import argparse

from libs.utils import get_logger
from libs.opts import StrToBoolAction
from libs.data_handler import WaveReader

logger = get_logger(__name__)


def run(args):
    wav_reader = WaveReader(args.wav_scp, sr=args.sr, normalize=False)
    with open(args.dur_scp, "w") as dur_scp:
        n = 0
        for key, wav in wav_reader:
            n += 1
            dur = wav.shape[-1]
            if args.output == "time":
                dur = float(dur) / args.sr
                dur_scp.write(f"{key} {dur:.2f}\n")
            else:
                dur_scp.write(f"{key} {dur:d}\n")
            if n % 50 == 0:
                dur_scp.flush()
                logger.info(f"Processed {n} utterances...")
    logger.info(f"Processed {len(wav_reader)} utterances done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to show the duration of the audio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("wav_scp",
                        type=str,
                        help="Rspecifier for source audio")
    parser.add_argument("dur_scp", type=str, help="Output duration script")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample rate of the audio")
    parser.add_argument("--output",
                        choices=["time", "number"],
                        default="number",
                        help="Show duration using seconds (time) "
                        "or number of samples (number)")
    args = parser.parse_args()
    run(args)