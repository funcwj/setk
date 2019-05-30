#!/usr/bin/env python

# wujian@2018
"""
Compute score for speaker varification tasks
"""

import argparse
import numpy as np

from libs.data_handler import NumpyReader, ScriptReader, parse_scps
from libs.utils import get_logger

logger = get_logger(__name__)


def run(args):
    utt2spk = parse_scps(args.utt2spk)

    def Reader(scp, t):
        return NumpyReader(scp) if t == "numpy" else ScriptReader(scp,
                                                                  matrix=False)

    spks_reader = Reader(args.spks_scp, args.type)
    spks_keys, spks_embs = [], []
    for spkid, spkvec in spks_reader:
        spks_keys.append(spkid)
        spks_embs.append(spkvec)
    spks_mat = np.stack(spks_embs)
    if args.normalize:
        spks_mat = np.linalg.norm(spks_mat, axis=1, ord=2, keepdims=True)
    logger.info("Load {:d} speakers from enrollment embeddings".format(
        len(spks_keys)))

    eval_reader = Reader(args.eval_scp, args.type)
    for uttid, uttvec in eval_reader:
        spkid = utt2spk[uttid]
        if args.normalize:
            uttvec = uttvec / np.linalg.norm(uttvec)
        if spkid not in spks_keys:
            raise RuntimeError(
                "Seems speaker {} do not exist in enrollment set".format(
                    spkid))
        # using dot product, because embeddings has been normalized
        # 1 x N
        score_mat = uttvec @ np.transpose(spks_mat)
        for index, cmpid in enumerate(spks_keys):
            print("{:.2f} {}".format(
                score_mat[index], "target" if cmpid == spkid else "nontarget"))
    logger.info("Compute scores for {:d} utterances done".format(
        len(eval_reader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to compute scores between candidate embeddings "
        "and registered ones, output results to stdout, which could "
        "be used to compute eer using compute-eer in kaldi.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("spks_scp",
                        type=str,
                        help="Embedding rspecifier computed "
                        "from enrollment utterances")
    parser.add_argument("eval_scp",
                        type=str,
                        help="Embedding rspecifier to evaluate perfermance")
    parser.add_argument("--utt2spk",
                        type=str,
                        required=True,
                        help="Rspecifier for utterance to speaker map")
    parser.add_argument("--vector-type",
                        dest="type",
                        type=str,
                        choices=["kaldi", "numpy"],
                        default="kaldi",
                        help="Storage format for embeddings")
    parser.add_argument("--normalize",
                        action="store_true",
                        help="If true, normalize embeddings "
                        "before compute dot product")
    args = parser.parse_args()
    run(args)