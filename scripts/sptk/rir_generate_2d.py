#!/usr/bin/env python

# wujian@2019
"""
Generate scripts for RIR simulation which use
    1) rir-simulate (see src/rir-simulate.cc)
    2) pyrirgen (see https://github.com/Marvin182/rir-generator)
"""
import os
import json
import shutil
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt

from libs.scheduler import run_command
from libs.utils import get_logger, make_dir, write_wav
from libs.opts import StrToFloatTupleAction, str_to_float_tuple

try:
    import pyrirgen
    pyrirgen_available = True
except ImportError:
    pyrirgen_available = False

logger = get_logger(__name__)


class UniformSampler(object):
    """
    A simple uniform sampler
    """

    def __init__(self, a, b):
        self.min, self.max = a, b

    def sample(self):
        return random.uniform(self.min, self.max)


class Room(object):
    """
    Room instance
    """

    def __init__(self, l, w, h, rt60=None, refl=None):
        self.size = (l, w, h)
        self.beta = rt60 if rt60 is not None else [refl] * 6
        self.memo = "{}={:.2f}".format("RT60" if rt60 is not None else "Refl",
                                       rt60 if rt60 is not None else refl)

    def set_mic(self, topo, center):
        """
        Place microphone array
        topo: tuple like [(x1, y1), ...]
        center: center 3D postion for microphone array
        """
        Mx, My, Mz = center
        self.rpos = [(Mx + x, My + y, Mz) for (x, y) in topo]
        self.topo = topo
        self.rcen = (Mx, My)

    def set_spk(self, pos):
        """
        Place sound source (speaker)
        """
        self.spos = pos

    def conf(self):
        """
        Return configure of room (exclude sound sources)
        """
        Rf = lambda f: round(f, 3)
        return {
            "beta": [Rf(f) for f in self.beta]
            if isinstance(self.beta, list) else Rf(self.beta),
            "receiver_location": [tuple(Rf(n) for n in p) for p in self.rpos],
            "room_size": [Rf(n) for n in self.size],
            "receiver_geometric":
            self.topo
        }

    def plot(self, scfg, fig, room_id):
        """
        Visualize microphone array and speakers in current room
        """
        ax = plt.subplot(111)
        ax.set_aspect("equal", "box")
        # constraint length and width
        l, w, _ = self.size
        plt.xlim((0, l))
        plt.ylim((0, w))
        # draw microphone array
        plt.plot([p[0] for p in self.rpos], [p[1] for p in self.rpos], "k.")
        plt.plot([self.rcen[0]], [self.rcen[1]], "r+")
        # draw each speaker
        for idx, cfg in enumerate(scfg):
            x, y, _ = cfg["pos"]
            plt.plot([x], [y], "k+")
            # "({:.2f}, {:.2f})".format(cfg["doa"], cfg["dst"])
            plt.annotate(str(idx + 1), xy=(x, y), xytext=(x + 0.1, y + 0.1))
        plt.xlabel("Length({:.2f}m)".format(l))
        plt.ylabel("Width({:.2f}m)".format(w))
        plt.title("{0}({1})".format(room_id, self.memo))
        plt.savefig(fig)
        plt.close()

    def rir(self, fname, fs=16000, rir_nsamps=4096, v=340):
        """
        Generate rir for current settings
        """
        if shutil.which("rir-simulate"):
            # format float
            ffloat = lambda f: "{:.3f}".format(f)
            # location for each microphone
            loc_for_each_channel = [
                ",".join(map(ffloat, p)) for p in self.rpos
            ]
            beta = ",".join(map(ffloat, self.beta)) if isinstance(
                self.beta, list) else round(self.beta, 3)
            run_command(
                "rir-simulate --sound-velocity={v} --samp-frequency={sample_rate} "
                "--hp-filter=true --number-samples={rir_samples} --beta={beta} "
                "--room-topo={room_size} --receiver-location=\"{receiver_location}\" "
                "--source-location={source_location} {dump_dest}".format(
                    v=v,
                    sample_rate=fs,
                    rir_samples=rir_nsamps,
                    room_size=",".join(map(ffloat, self.size)),
                    beta=beta,
                    receiver_location=";".join(loc_for_each_channel),
                    source_location=",".join(map(ffloat, self.spos)),
                    dump_dest=fname))
        elif pyrirgen_available:
            rir = pyrirgen.generateRir(self.size,
                                       self.spos,
                                       self.rpos,
                                       soundVelocity=v,
                                       fs=fs,
                                       nDim=3,
                                       nSamples=rir_nsamps,
                                       nOrder=-1,
                                       reverbTime=self.beta,
                                       micType="omnidirectional",
                                       isHighPassFilter=True)
            if isinstance(rir, list):
                rir = np.stack(rir)
            write_wav(fname, rir, fs=fs)
        else:
            raise RuntimeError("Both rir-simulate and pyrirgen unavailable")


class RoomGenerator(object):
    """
    Room generator
    """

    def __init__(self, rt60_opt, absc_opt, room_dim):
        """
        rt60_opt: "" or "a,b", higher priority than absc_opt
        absc_opt: tuple like (a,b)
        room_dim: str like "a,b;c,d;e,d"
        """
        self.rt60_opt = rt60_opt
        if not rt60_opt:
            self.absc = UniformSampler(*absc_opt)
        else:
            rt60_r = str_to_float_tuple(rt60_opt)
            self.rt60 = UniformSampler(*rt60_r)
        dim_range = [str_to_float_tuple(t) for t in room_dim.split(";")]
        if len(dim_range) != 3:
            raise RuntimeError(
                "Wrong format with --room-dim={}".format(room_dim))
        self.dim_sampler = [UniformSampler(*c) for c in dim_range]

    def generate(self, v=340):
        # (l, w, h)
        (l, w, h) = (s.sample() for s in self.dim_sampler)
        if self.rt60_opt:
            # no reflection is ok
            if self.rt60.max == 0:
                return Room(l, w, h, rt60=0)
            else:
                # check rt60 here
                S, V = l * w * h, (l * w + l * h + w * h) * 2
                # sabine formula
                rt60_min = 24 * V * np.log(10) / (v * S)
                if rt60_min >= self.rt60.max:
                    return None
                    # raise RuntimeError(
                    #     "Configuration error in rt60: {}, minimum {:.2f} "
                    #     "required with ({:.2f}x{:.2f}x{:.2f})".format(
                    #         self.rt60_opt, rt60_min, l, w, h))
                else:
                    rt60 = random.uniform(rt60_min, self.rt60.max)
                    return Room(l, w, h, rt60=rt60)
        else:
            absc = self.absc.sample()
            return Room(l, w, h, refl=np.sqrt(1 - absc))


class RirSimulator(object):
    """
    RIR simulator
    """

    def __init__(self, args):
        # make dump dir
        make_dir(args.dump_dir)
        self.rirs_cfg = []
        self.room_generator = RoomGenerator(args.rt60, args.abs_range,
                                            args.room_dim)
        self.mx, self.my = args.array_relx, args.array_rely
        self.array_topo = [
            str_to_float_tuple(t) for t in args.array_topo.split(";")
        ]
        self.args = args

    def _place_mic(self, room):
        x, y, _ = room.size
        # sample array location
        # (mx, my) center postion of array
        mx = random.uniform(*(x * v for v in self.mx))
        my = random.uniform(*(y * v for v in self.my))
        mz = random.uniform(*self.args.array_height)
        # place array
        room.set_mic(self.array_topo, (mx, my, mz))
        return (mx, my), room

    def _max_src_dist(self, rpos_2d, room_size_2d):
        mx, my = rpos_2d
        rx, ry = room_size_2d
        bound = [(0, 0), (0, ry), (rx, 0), (rx, ry)]
        dist = [((mx - x)**2 + (my - y)**2)**0.5 for (x, y) in bound]
        return max(dist)

    def _place_spk(self, center, room):
        num_rirs = self.args.num_rirs
        done, ntry = 0, 0
        mx, my = center
        rx, ry, rz = room.size
        max_retry = self.args.retry * num_rirs
        stats = []

        min_src_dist, max_src_dist = args.src_dist
        max_src_dist = min(max_src_dist, self._max_src_dist((mx, my),
                                                            (rx, ry)))

        Rf = lambda f: round(f, 3)
        while True:
            ntry += 1
            if ntry > max_retry:
                break
            sz = random.uniform(*args.speaker_height)
            if sz >= rz:
                continue

            # speaker distance
            dst = random.uniform(min_src_dist, max_src_dist)
            # sample from 0-360
            doa = random.uniform(0, np.pi * 2)

            sx = mx + np.cos(doa) * dst
            sy = my + np.sin(doa) * dst

            # check speaker location
            if 0 > sx or sx > rx or 0 > sy or sy > ry:
                continue
            done += 1
            stat = {
                "pos": (Rf(sx), Rf(sy), Rf(sz)),
                "doa": Rf(doa * 180 / np.pi),
                "dst": Rf(dst)
            }
            stats.append(stat)
            if done == num_rirs:
                break
        logger.info("try/done = {:d}/{:d}".format(ntry, done))
        return done == num_rirs, stats

    def run_for_instance(self, room_id):
        room = None
        while not room:
            room = self.room_generator.generate(v=self.args.speed)
        rpos, room = self._place_mic(room)
        succ, scfg = self._place_spk(rpos, room)
        if succ:
            rcfg = room.conf()
            for idx, stat in enumerate(scfg):
                # place spk and generate rir
                room.set_spk(stat["pos"])
                rir_loc = "{0}/Room{1}-{2}.wav".format(self.args.dump_dir,
                                                       room_id, idx + 1)
                room.rir(rir_loc,
                         fs=self.args.sample_rate,
                         rir_nsamps=self.args.rir_samples,
                         v=self.args.speed)
                scfg[idx]["loc"] = rir_loc
            # plot room
            room.plot(scfg, "{0}/Room{1}.png".format(self.args.dump_dir,
                                                     room_id),
                      "Room{:d}".format(room_id))
            rcfg["spk"] = scfg
            self.rirs_cfg.append(rcfg)
        return succ

    def run(self):
        num_rooms = self.args.num_rooms
        max_retry = self.args.retry * num_rooms
        done, ntry = 0, 0
        while True:
            ntry += 1
            if ntry > max_retry:
                break
            succ = self.run_for_instance(done + 1)
            if succ:
                done += 1
            if done == num_rooms:
                break
        # dump rir configurations
        with open(os.path.join(args.dump_dir, "rir.json"), "w") as f:
            json.dump(self.rirs_cfg, f, indent=2)
        logger.info("Generate {:d} rirs, {:d} rooms done, "
                    "retry = {:d}".format(self.args.num_rirs * num_rooms, done,
                                          ntry))


def run(args):
    simulator = RirSimulator(args)
    simulator.run()


"""
Run egs:
$cmd JOB=1:$nj ./exp/rir_simu/rir_generate_2d.JOB.log \
  ./scripts/sptk/rir_generate_2d.py \
    --num-rirs $num_rirs \
    --dump-dir $dump_dir/JOB \
    --array-height "1.2,1.8" \
    --room-dim "4,10\;4,10\;2,4" \
    --rt60 "0.2,0.5" \
    --array-relx "0.3,0.7" \
    --array-rely "0.3,0.7" \
    --speaker-height "1,2" \
    --source-distance "1,4" \
    --rir-samples 4096 \
    --dump-cfg \
    $num_room
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to generate single/multi-channel RIRs"
        "(using rir-simulate or pyrirgen from https://github.com/Marvin182/rir-generator)"
        "In this command, we will simulate several rirs for each room, which is "
        "configured using --num-rirs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("num_rooms",
                        type=int,
                        help="Total number of rooms to simulate")
    parser.add_argument("--num-rirs",
                        type=int,
                        default=1,
                        help="Number of rirs to simulate for each room")
    parser.add_argument("--dump-cfg",
                        action="store_true",
                        help="If true, dump rir configures out in json format")
    parser.add_argument("--rir-samples",
                        type=int,
                        default=8000,
                        help="Number samples of simulated rir")
    parser.add_argument("--sample-rate",
                        type=int,
                        default=16000,
                        help="Sample rate of simulated signal")
    parser.add_argument("--dump-dir",
                        type=str,
                        default="rir",
                        help="Directory to dump generated rirs")
    parser.add_argument("--room-dim",
                        type=str,
                        default="7,10;7,10;3,4",
                        help="Constraint for room length/width/height, "
                        "separated by semicolon")
    parser.add_argument("--array-height",
                        action=StrToFloatTupleAction,
                        default=(1, 2),
                        help="Range of array's height")
    parser.add_argument("--array-relx",
                        action=StrToFloatTupleAction,
                        default=(0.4, 0.6),
                        help="Area of room to place microphone array randomly"
                        "(relative values to room's length)")
    parser.add_argument("--array-rely",
                        action=StrToFloatTupleAction,
                        default=(0.05, 0.1),
                        help="Area of room to place microphone array randomly"
                        "(relative values to room's width)")
    parser.add_argument("--speaker-height",
                        action=StrToFloatTupleAction,
                        default=(1.6, 2),
                        help="Range of speaker's height")
    parser.add_argument("--array-topo",
                        type=str,
                        default="0,0;0.06,0;0.03,0.05196;"
                        "-0.03,0.05196;-0.06,0;"
                        "-0.03,-0.05196;0.03,0.05196",
                        help="Linear topology for microphone arrays.")
    parser.add_argument("--absorption-coefficient-range",
                        action=StrToFloatTupleAction,
                        dest="abs_range",
                        default=(0.2, 0.8),
                        help="Range of absorption coefficient "
                        "of the room material. Absorption coefficient "
                        "is located between 0 and 1, if a material "
                        "offers no reflection, the absorption "
                        "coefficient is close to 1.")
    parser.add_argument("--rt60",
                        type=str,
                        default="0.2,0.7",
                        help="Range of RT60, this option "
                        "has higher priority than "
                        "--absorption-coefficient-range")
    parser.add_argument("--sound-speed",
                        type=float,
                        dest="speed",
                        default=340,
                        help="Speed of sound")
    parser.add_argument("--retry",
                        type=int,
                        default=5,
                        help="Max number of times tried to generate rirs "
                        "for a specific room (retry * num_rirs)")
    parser.add_argument("--source-distance",
                        action=StrToFloatTupleAction,
                        dest="src_dist",
                        default=(1, 3),
                        help="Range of distance between "
                        "microphone arrays and speakers")
    args = parser.parse_args()
    run(args)
