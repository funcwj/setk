#!/usr/bin/env python

# wujian@2018
"""
Generate scripts for RIR simulation which use rir-simulate(see src/rir-simulate.cc)
"""
import argparse
import random
import os
import tqdm

import numpy as np

from libs.scheduler import run_command
from libs.utils import get_logger

logger = get_logger(__name__)


def str_to_float_tuple(string, sep=","):
    tokens = string.split(sep)
    if len(tokens) == 1:
        raise ValueError("Get only one token by sep={0}, string={1}".format(
            sep, string))
    floats = map(float, tokens)
    return tuple(floats)


class StrToFloatTupleAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            setattr(namespace, self.dest, str_to_float_tuple(values))
        except ValueError:
            raise Exception("Unknown value {0} for --{1}".format(
                values, self.dest))


def parse_config(args):
    args_dict = dict()
    # process t60
    args_dict["t60_min"], args_dict["t60_max"] = args.t60_range
    # process source-distance
    args_dict["dst_min"], args_dict["dst_max"] = args.src_dist
    # process array-topo
    args_dict["topo"] = args.array_topo
    # process room-dim
    assert args.room_dim, "--room-dim could not be None"
    tokens = args.room_dim.split(";")
    if len(tokens) != 3:
        raise ValueError(
            "--room-dim must be set for length/width/height respectively")
    for index, name in enumerate(["Rx", "Ry", "Rz"]):
        min_, max_ = str_to_float_tuple(tokens[index])
        args_dict["{}_min".format(name)] = min_
        args_dict["{}_max".format(name)] = max_
    # process --array-height
    args_dict["Mz_min"], args_dict["Mz_max"] = args.array_height
    # process --array-area
    assert args.array_area, "--array-area could not be None"
    tokens = args.array_area.split(";")
    if len(tokens) != 2:
        raise ValueError(
            "--array-area should be set for length/width respectively")
    for index, name in enumerate(["Mx", "My"]):
        min_, max_ = str_to_float_tuple(tokens[index])
        args_dict["{}_min".format(
            name)] = min_ * args_dict["Rx_min" if index == 0 else "Ry_min"]
        args_dict["{}_max".format(
            name)] = max_ * args_dict["Rx_max" if index == 0 else "Ry_max"]
    # process --speaker-height
    args_dict["Sz_min"], args_dict["Sz_max"] = args.speaker_height
    # process --array-topo
    args_dict["topo"] = args.array_topo
    return args_dict


def run(args):
    conf = parse_config(args)

    def sample_value(conf, key):
        return random.uniform(conf["{}_min".format(key)],
                              conf["{}_max".format(key)])

    def format_float(float_n):
        return "{:.3f}".format(float_n)

    if not os.path.exists(args.dump_dir):
        os.makedirs(args.dump_dir)

    logger.info(
        "This command will generate {:d} rirs in total, {:d} for each room".
        format(args.num_rirs * args.num_rooms, args.num_rirs))

    for room_id in tqdm.trange(args.num_rooms, desc="Finished Rooms"):
        done_cur_room = 0

        while True:
            if done_cur_room == args.num_rirs:
                break
            # generate configure for current room
            Rx, Ry, Rz = sample_value(conf, "Rx"), sample_value(
                conf, "Ry"), sample_value(conf, "Rz")
            # microphone location
            Mx, My = sample_value(conf, "Mx"), sample_value(conf, "My")
            # speaker location
            dst = sample_value(conf, "dst")
            # sample from 0-180
            doa = random.uniform(0, np.pi)
            Sx = Mx + np.cos(doa) * dst
            Sy = My + np.sin(doa) * dst

            # check speaker location
            if 0 > Sx or Sx > Rx or 0 > Sy or Sy > Ry:
                continue
            # speaker and microphone height
            Sz = sample_value(conf, "Sz")
            Mz = sample_value(conf, "Mz")
            # check peaker and microphone height
            if Sz > Rz or Mz > Rz:
                continue

            done_cur_room += 1
            source_location = ",".join(map(format_float, [Sx, Sy, Sz]))
            room_size = ",".join(map(format_float, [Rx, Ry, Rz]))
            # center position
            Mc = (conf["topo"][-1] - conf["topo"][0]) / 2
            # coordinate of each channel
            loc_for_each_channel = [
                ",".join(map(format_float, [Mx - Mc + x, My, Mz]))
                for x in conf["topo"]
            ]
            # T60
            t60 = "{:.3f}".format(sample_value(conf, "t60"))

            rir_path = "{dir}/Room{room_id}-T60-{t60}-DoA-{doa}-Dst{dst}.wav".format(
                dir=args.dump_dir,
                room_id=room_id,
                t60=t60,
                doa=int(doa * 180 / np.pi),
                dst="{:.2f}".format(dst))
            # generate rir using rir-simulate command
            run_command(
                "rir-simulate --sound-velocity=340 --samp-frequency={sample_rate} "
                "--hp-filter=true --number-samples={rir_samples} --beta={t60} "
                "--room-topo={room_size} --receiver-location=\"{receiver_location}\" "
                "--source-location={source_location} {rir}".format(
                    sample_rate=args.sample_rate,
                    rir_samples=args.rir_samples,
                    room_size=room_size,
                    t60=t60,
                    receiver_location=";".join(loc_for_each_channel),
                    source_location=source_location,
                    rir=rir_path))
    logger.info("Generate {:d} rirs in total done".format(
        args.num_rirs * args.num_rooms))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to generate single/multi-channel RIRs by calling rir-simulate"
        "(Please export $PATH to enable rir-simulate found by system environment)"
        "In this command, we will simulate several rirs for each room, which is "
        "configured using --num-rirs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "num_rooms", type=int, help="Total number of rooms to simulate")
    parser.add_argument(
        "--num-rirs",
        type=int,
        default=1,
        dest="num_rirs",
        help="Number of rirs to simulate for each room")
    parser.add_argument(
        "--rir-samples",
        type=int,
        default=4096,
        dest="rir_samples",
        help="Number samples of simulated rir")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        dest="sample_rate",
        help="Sample rate of simulated signal")
    parser.add_argument(
        "--dump-dir",
        type=str,
        dest="dump_dir",
        default="",
        help="Directory to dump generated rirs")
    parser.add_argument(
        "--room-dim",
        type=str,
        dest="room_dim",
        default="7,10;7,10;3,4",
        help="Constraint for room length/width/height, separated by semicolon")
    parser.add_argument(
        "--array-height",
        action=StrToFloatTupleAction,
        dest="array_height",
        default=(1, 2),
        help="Range of array's height")
    parser.add_argument(
        "--speaker-height",
        action=StrToFloatTupleAction,
        dest="speaker_height",
        default=(1.6, 2),
        help="Range of speaker's height")
    parser.add_argument(
        "--array-area",
        type=str,
        dest="array_area",
        default="0.4,0.6;0,0.1",
        help="Area of room to place microphone arrays randomly"
        "(relative values to room's length and width)")
    parser.add_argument(
        "--array-topo",
        action=StrToFloatTupleAction,
        dest="array_topo",
        default=(0, 0.1, 0.2, 0.3),
        help="Linear topology for microphone arrays.")
    parser.add_argument(
        "--t60-range",
        action=StrToFloatTupleAction,
        dest="t60_range",
        default=(0.2, 0.7),
        help="Range of T60 values.")
    parser.add_argument(
        "--source-distance",
        action=StrToFloatTupleAction,
        dest="src_dist",
        default=(2, 3),
        help="Range of distance between microphone arrays and speakers")
    args = parser.parse_args()
    run(args)