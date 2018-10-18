#!/usr/bin/env python

# wujian@2018
"""
Generate scripts for RIR simulation.
"""
import argparse
import random
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

    def random_value(conf, key):
        return random.uniform(conf["{}_min".format(key)],
                              conf["{}_max".format(key)])

    for _ in range(args.num_rirs):
        room_size = "{:.3f},{:.3f},{:.3f}".format(
            random_value(conf, "Rx"), random_value(conf, "Ry"),
            random_value(conf, "Rz"))
        # microphone location
        Mx, My = random_value(conf, "Mx"), random_value(conf, "My")
        # speaker location
        dst = random_value(conf, "dst")
        doa = random.uniform(0, np.pi)
        Sx = Mx + np.cos(doa) * dst
        Sy = My + np.sin(doa) * dst
        source_location = "{:.3f},{:.3f},{:.3f}".format(
            Sx, Sy, random_value(conf, "Sz"))
        print(
            "# Location(M): ({:.2f}, {:.2f}), Location(S): ({:.2f}, {:.2f}), DoA: {:d}"
            .format(Mx, My, Sx, Sy, int(doa * 180 / np.pi)))

        Mc = (conf["topo"][-1] - conf["topo"][0]) / 2
        Mz = random_value(conf, "Mz")
        loc_for_each_channel = [
            "{:.3f},{:.3f},{:.3f}".format(Mx - Mc + x, My, Mz)
            for x in conf["topo"]
        ]

        print(
            "rir-simulate --sound-velocity=340 --samp-frequency=16000 --hp-filter=true "
            "--number-samples=2048 --beta={t60} --room-topo={room_size} "
            "--receiver-location=\"{receiver_location}\" --source-location={source_location} "
            "{dir}/T60-{t60}-DoA-{doa}-Dst{dst}.wav".format(
                room_size=room_size,
                t60="{:.3f}".format(random_value(conf, "t60")),
                receiver_location=";".join(loc_for_each_channel),
                source_location=source_location,
                dir=args.dst_dir,
                doa=int(doa * 180 / np.pi),
                dst=dst))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to generate scripts for single/multi-channel RIRs simulation. (Note: print to stdout)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "num_rirs", type=int, help="Total number of rirs to generate")
    parser.add_argument(
        "dst_dir", type=str, help="Location to dump simulated rirs")
    parser.add_argument(
        "--room-dim",
        type=str,
        dest="room_dim",
        default="5,10;5,10;3,4",
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
        default="0.4,0.6;0.4,0.6",
        help=
        "Range of room to place microphone arrays(relative to room's length and width)"
    )
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
        default="0.75,2",
        help="Range of distance between microphone arrays and speakers")
    args = parser.parse_args()
    run(args)