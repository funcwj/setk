#!/usr/bin/env python

# wujian@2018

import os
import argparse

import torch as th
import numpy as np

from nnet import Nnet
from kaldi_python_io import ScriptReader

from libs.utils import load_json, get_logger, make_dir
from libs.dataset import Processor

logger = get_logger(__name__)


class NnetComputer(object):
    def __init__(self, cpt_dir, gpuid):
        # load nnet conf
        nnet_conf = load_json(cpt_dir, "mdl.json")
        nnet = Nnet(**nnet_conf)
        # load checkpoint
        cpt_fname = os.path.join(cpt_dir, "best.pt.tar")
        cpt = th.load(cpt_fname, map_location="cpu")
        nnet.load_state_dict(cpt["model_state_dict"])
        logger.info("Load checkpoint from {}, epoch {:d}".format(
            cpt_fname, cpt["epoch"]))
        # handle device
        self.device = th.device(
            "cpu" if gpuid < 0 else "cuda:{}".format(gpuid))
        self.nnet = nnet.to(self.device) if gpuid >= 0 else nnet
        # set eval model
        self.nnet.eval()

    def compute(self, feats):
        def tensor(mat):
            return th.tensor(mat, dtype=th.float32, device=self.device)

        with th.no_grad():
            feats = tensor(feats)
            spk_masks = self.nnet(feats, train=False)
            return [m.detach().cpu().numpy() for m in spk_masks]


def run(args):
    computer = NnetComputer(args.checkpoint, args.gpu)
    num_done = 0
    feats_conf = load_json(args.checkpoint, "feats.json")
    spectra = Processor(args.spectra, **feats_conf)
    spatial = ScriptReader(args.spatial) if args.spatial else None

    for key, feats in spectra:
        logger.info("Compute on utterance {}...".format(key))
        if spatial:
            spa = spatial[key]
            feats = np.hstack([feats, spa])
        spk_masks = computer.compute(feats)
        for i, m in enumerate(spk_masks):
            fdir = os.path.join(args.dump_dir, "spk{:d}".format(i + 1))
            make_dir(fdir)
            np.save(os.path.join(fdir, key), m)
        num_done += 1
    logger.info("Compute over {:d} utterances".format(num_done))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to compute speaker masks from uPIT models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoint", type=str, help="Directory of checkpoint")
    parser.add_argument(
        "--spectra",
        type=str,
        required=True,
        help="Script for input spectra features")
    parser.add_argument(
        "--spatial", type=str, default="", help="Script for spatial features")
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="GPU-id to offload model to, -1 means running on CPU")
    parser.add_argument(
        "--dump-dir",
        type=str,
        default="masks",
        help="Directory to dump masks out")
    args = parser.parse_args()
    run(args)