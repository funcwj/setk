#!/usr/bin/env python

# wujian@2020

import json
import uuid
import random
import argparse

import numpy as np

from libs.data_handler import ScpReader
from libs.sampler import UniformSampler


class UtteranceSampler(object):
    def __init__(self, loc_and_dur):
        toks = loc_and_dur.split(",")
        if len(toks) != 2:
            raise RuntimeError(f"format error: {loc_and_dur}")
        loc, dur = toks
        self.loc = ScpReader(loc)
        self.dur = ScpReader(dur, value_processor=lambda x: int(x))

    def sample(self):
        key, path = self.loc.sample(1)
        if key not in self.dur:
            raise RuntimeError(f"Missing key: {key} in duration scripts")
        return path, self.dur[key]


def run(args):

    random.seed(args.seed)
    np.random.seed(args.seed)

    iso_snr_sampler = UniformSampler(args.isotropic_noise_snr)
    ptn_snr_sampler = UniformSampler(args.point_noise_snr)
    factor_sampler = UniformSampler(args.norm_factor)

    with open(args.rir_cfg, "r") as rir_cfg:
        room_rir = json.load(rir_cfg)

    src_sampler = UtteranceSampler(args.src_spk)
    ptn_sampler = UtteranceSampler(args.point_noise)
    iso_sampler = None
    if args.isotropic_noise:
        iso_sampler = UtteranceSampler(args.isotropic_noise)

    with open(args.simu_cfg, "w") as simu_cfg:
        for _ in range(args.num_utts):
            key = str(uuid.uuid4())
            factor = factor_sampler.sample()
            # path, duration
            spk_path, spk_dur = src_sampler.sample()
            ptn_path, ptn_dur = ptn_sampler.sample()
            # room
            room = random.sample(room_rir, 1)[0]
            rirs = random.sample(room["spk"], 2)
            # ptn_snr
            ptn_snr = ptn_snr_sampler.sample()

            if ptn_dur < spk_dur:
                ptn_beg = random.randint(0, spk_dur - ptn_dur)
            else:
                ptn_beg = 0

            cmd_args = f"--dump-ref-dir {args.simu_dir} --src-spk {spk_path}" + \
                f"--src-rir {rirs[0]} --src-begin 0 " + \
                f"--point-noise {ptn_path} --point-noise-rir {rirs[1]}" + \
                f"--point-noise-snr {ptn_snr:.3f} --sr {args.sr} --norm-factor {factor:.2f}" \
                f"----point-noise-begin {ptn_beg}"

            if iso_sampler:
                iso_path, iso_dur = iso_sampler.sample()
                iso_snr = iso_snr_sampler.sample()
                if iso_dur > spk_dur:
                    iso_beg = random.randint(0, iso_dur - spk_dur)
                else:
                    iso_beg = 0
                simu_cfg.write(
                    f"python wav_simulate.py {args.simu_dir}/clean/{key}.wav {cmd_args}"
                    +
                    f"--isotropic-noise {iso_path} --isotropic-noise-snr {iso_snr} "
                    + f"--isotropic-noise-begin {iso_beg}")
            else:
                simu_cfg.write(
                    f"python wav_simulate.py {args.simu_dir}/clean/{key}.wav {cmd_args}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to create noisy data simulation configurations, "
        "for wav_simulate.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    parser.add_argument("src_spk",
                        type=str,
                        help="Source speaker location & duration scripts")
    parser.add_argument("rir_cfg",
                        type=str,
                        help="RIR configurations (in json format). "
                        "See rir_generate_{1,2}d")
    parser.add_argument("simu_cfg",
                        type=str,
                        help="Output configuration of the data simulation")
    parser.add_argument("--num-utts",
                        type=int,
                        default=1000,
                        help="Number of the simulated utterances")
    parser.add_argument("--simu-dir",
                        type=str,
                        default="simu",
                        help="Simulation directory to dump data")
    parser.add_argument("--point-noise",
                        type=str,
                        required=True,
                        help="Pointsource noise location & duration scripts")
    parser.add_argument("--point-noise-snr",
                        type=str,
                        default="5,15",
                        help="SNR of the pointsource noises")
    parser.add_argument("--isotropic-noise",
                        type=str,
                        default="",
                        help="Isotropic noise location & duration scripts")
    parser.add_argument("--isotropic-noise-snr",
                        type=str,
                        default="10,20",
                        help="SNR of the isotropic noises")
    parser.add_argument("--dump-channel",
                        type=int,
                        default=-1,
                        help="Index of the channel to dump out (-1 means all)")
    parser.add_argument('--norm-factor',
                        type=float,
                        default="0.5,0.9",
                        help="Normalization factor of the final output")
    parser.add_argument("--seed",
                        type=int,
                        default=666,
                        help="Random seed to set")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Value of the sample rate")
    run(args)