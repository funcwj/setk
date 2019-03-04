#!/usr/bin/env python

# wujian@2018

import os
import pprint
import argparse
import random

from nnet import UpitNet
from trainer import UpitTrainer
from utils import dump_json, get_logger
from dataset import make_pitloader
from conf import trainer_conf, nnet_conf, feats_conf, train_data, dev_data

logger = get_logger(__name__)


def run(args):
    nnet = UpitNet(**nnet_conf)

    trainer = UpitTrainer(
        nnet, gpuid=args.gpu, checkpoint=args.checkpoint, **trainer_conf)

    for conf, fname in zip([nnet_conf, feats_conf, trainer_conf],
                           ["mdl.json", "feats.json", "trainer.json"]):
        dump_json(conf, args.checkpoint, fname)

    feats_conf["shuf"] = True
    train_loader = make_pitloader(
        train_data["linear_x"],
        feats_conf,
        train_data,
        num_utts=args.batch_size)
    feats_conf["shuf"] = False
    dev_loader = make_pitloader(
        dev_data["linear_x"], feats_conf, dev_data, num_utts=args.batch_size)

    trainer.run(train_loader, dev_loader, num_epochs=args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to do permutate invariant training, "
        "auto configured from conf.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--gpu", type=int, default=0, help="Training on which GPUs")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="exp/upit/1a",
        help="Directory to dump models")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of utterances in each batch")
    args = parser.parse_args()
    logger.info("Arguments in command:\n{}".format(pprint.pformat(vars(args))))
    run(args)
