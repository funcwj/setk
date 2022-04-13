#!/usr/bin/env python

# wujian@2020

import argparse
import sys
import warnings
import wave


def ext_open(fd, mode):
    if mode not in ["r", "w"]:
        raise ValueError(f"Unsupported mode: {mode}")
    if fd == "-":
        return sys.stdout if mode == "w" else sys.stdin
    else:
        return open(fd, mode)


def run(args):
    prog_interval = 100
    done, total = 0, 0
    with ext_open(args.utt2dur, "w") as utt2dur:
        with ext_open(args.wav_scp, "r") as wav_scp:
            for raw_line in wav_scp:
                total += 1
                line = raw_line.strip()
                toks = line.split()
                if len(toks) != 2:
                    warnings.warn(f"Line format error: {line}")
                    continue
                done += 1
                key, path = toks
                with wave.open(path, "r") as wav:
                    dur = wav.getnframes()
                    if args.output == "time":
                        dur = float(dur) / wav.getframerate()
                if args.output == "time":
                    utt2dur.write(f"{key}\t{dur:.4f}\n")
                else:
                    utt2dur.write(f"{key}\t{dur:d}\n")
                if done % prog_interval == 0:
                    print(f"Processed {done} utterances...", flush=True)
    print(f"Processed {done} utterances done, total {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to generate duration of the wave. "
        "We avoid to read whole utterance as it may slow down the speed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("wav_scp", type=str, help="Input wave script")
    parser.add_argument("utt2dur", type=str, help="Output utt2dur file")
    parser.add_argument("--output",
                        type=str,
                        choices=["time", "sample"],
                        default="sample",
                        help="Output type of the script")
    args = parser.parse_args()
    run(args)
