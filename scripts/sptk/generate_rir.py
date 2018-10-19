#!/usr/bin/env python

# wujian@2018
"""
Generate scripts for RIR simulation which use rir-simulate(see src/rir-simulate.cc)
"""
import argparse
import random
import os

import numpy as np


def str_to_float_tuple(string, sep=","):
    tokens = string.split(sep)
    if len(tokens) == 1:
        raise ValueError("Get only one token by sep={0}, string={1}".format(
            sep, string))
    floats = map(float, tokens)
    return tuple(floats)


def parse_config(args):
    args_dict = dict()
    # process t60
    assert args.t60_range, "--t60-range could not be None"
    t60_min, t60_max = str_to_float_tuple(args.t60_range)
    args_dict["t60_min"] = t60_min
    args_dict["t60_max"] = t60_max
    # process source-distance
    assert args.src_dist, "--source-distance could not be None"
    dst_min, dst_max = str_to_float_tuple(args.src_dist)
    args_dict["dst_min"] = dst_min
    args_dict["dst_max"] = dst_max
    # process array-topo
    assert args.array_topo, "--array-topo could not be None"
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
    assert args.array_height, "--array-height could not be None"
    height_min, height_max = str_to_float_tuple(args.array_height)
    args_dict["Mz_min"] = height_min
    args_dict["Mz_max"] = height_max
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
    assert args.speaker_height, "--speaker-height could not be None"
    height_min, height_max = str_to_float_tuple(args.speaker_height)
    args_dict["Sz_min"] = height_min
    args_dict["Sz_max"] = height_max
    # process --array-topo
    assert args.array_topo, "--array-topo could not be None"
    args_dict["topo"] = str_to_float_tuple(args.array_topo)
    return args_dict


def run(args):
    conf = parse_config(args)
    print("#!/usr/bin/env bash")
    print("set -eu")
    print("[ -d {dir} ] && rm -rf {dir}/*.wav".format(dir=args.dump_dir))
    print("[ ! -d {dir} ] && mkdir -p {dir}".format(dir=args.dump_dir))

    def random_value(conf, key):
        return random.uniform(conf["{}_min".format(key)],
                              conf["{}_max".format(key)])

    num_done = 0
    while True:
        if num_done == args.num_rirs:
            break
        Rx, Ry, Rz = random_value(conf, "Rx"), random_value(
            conf, "Ry"), random_value(conf, "Rz")
        # microphone location
        Mx, My = random_value(conf, "Mx"), random_value(conf, "My")
        # speaker location
        dst = random_value(conf, "dst")
        doa = random.uniform(0, np.pi)
        Sx = Mx + np.cos(doa) * dst
        Sy = My + np.sin(doa) * dst

        # check speaker location
        if 0 > Sx or Sx > Rx or 0 > Sy or Sy > Ry:
            continue
        # speaker and microphone height
        Sz = random_value(conf, "Sz")
        Mz = random_value(conf, "Mz")
        # check peaker and microphone height
        if Sz > Rz or Mz > Rz:
            continue

        num_done += 1
        source_location = "{:.3f},{:.3f},{:.3f}".format(Sx, Sy, Sz)
        room_size = "{:.3f},{:.3f},{:.3f}".format(Rx, Ry, Rz)
        print("# Room: {:.2f} x {:.2f}, Location(M): ({:.2f}, {:.2f}), "
              "Location(S): ({:.2f}, {:.2f}), DoA: {:d}".format(
                  Rx, Ry, Mx, My, Sx, Sy, int(doa * 180 / np.pi)))
        # center position
        Mc = (conf["topo"][-1] - conf["topo"][0]) / 2
        loc_for_each_channel = [
            "{:.3f},{:.3f},{:.3f}".format(Mx - Mc + x, My, Mz)
            for x in conf["topo"]
        ]

        print(
            "rir-simulate --sound-velocity=340 --samp-frequency={sample_rate} --hp-filter=true "
            "--number-samples={rir_samples} --beta={t60} --room-topo={room_size} "
            "--receiver-location=\"{receiver_location}\" --source-location={source_location} "
            "{dir}/T60-{t60}-DoA-{doa}-Dst{dst}.wav".format(
                sample_rate=args.sample_rate,
                rir_samples=args.rir_samples,
                room_size=room_size,
                t60="{:.3f}".format(random_value(conf, "t60")),
                receiver_location=";".join(loc_for_each_channel),
                source_location=source_location,
                dir=args.dump_dir,
                doa=int(doa * 180 / np.pi),
                dst="{:.2f}".format(dst)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to generate scripts for single/multi-channel RIRs simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "num_rirs", type=int, help="Total number of rirs to generate")
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
        type=str,
        dest="array_height",
        default="1,2",
        help="Range of array's height")
    parser.add_argument(
        "--speaker-height",
        type=str,
        dest="speaker_height",
        default="1.6,2",
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
        type=str,
        dest="array_topo",
        default="0,0.037,0.113,0.226",
        help="Linear topology for microphone arrays.")
    parser.add_argument(
        "--t60-range",
        type=str,
        dest="t60_range",
        default="0.2,0.7",
        help="Range of T60 values.")
    parser.add_argument(
        "--source-distance",
        type=str,
        dest="src_dist",
        default="2,3",
        help="Range of distance between microphone arrays and speakers")
    args = parser.parse_args()
    run(args)