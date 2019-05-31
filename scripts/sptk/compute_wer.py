#!/usr/bin/env python

# wujian@2019

import argparse

from tqdm import tqdm

from libs.data_handler import Reader as BaseReader
from libs.metric import permute_ed


class TransReader(object):
    """
    Class to handle single/multi-speaker transcriptions
    """

    def __init__(self, text):
        self.text_reader = [
            BaseReader(t, num_tokens=-1, restrict=False)
            for t in text.split(",")
        ]

    def __len__(self):
        return len(self.text_reader)

    def __getitem__(self, key):
        return [reader[key] for reader in self.text_reader]

    def __iter__(self):
        ref = self.text_reader[0]
        for key in ref.index_keys:
            yield key, [reader[key] for reader in self.text_reader]


def run(args):
    hyp_reader = TransReader(args.hyp)
    ref_reader = TransReader(args.ref)
    if len(hyp_reader) != len(ref_reader):
        raise RuntimeError(
            "Looks number of speakers do not match in hyp & ref")
    each_utt = open(args.per_utt, "w") if args.per_utt else None

    err = 0
    tot = 0
    for key, hyp in tqdm(hyp_reader):
        ref = ref_reader[key]
        dst = permute_ed(hyp, ref)
        ref_len = sum([len(r) for r in ref])
        if each_utt:
            if ref_len != 0:
                each_utt.write("{}\t{:.3f}\n".format(key, dst / ref_len))
            else:
                each_utt.write("{}\tINF\n".format(key))
        err += dst
        tot += ref_len
    print("WER: {:.2f}%".format(err * 100 / tot))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to compute wer (edit/levenshtein distance), "
        "accepting text following Kaldi's format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("hyp",
                        type=str,
                        help="Hypothesis transcripts "
                        "(multi-speakers need split by ',')")
    parser.add_argument("ref",
                        type=str,
                        help="References transcripts "
                        "(multi-speakers need split by ',')")
    parser.add_argument("--per-utt",
                        type=str,
                        default="",
                        help="If assigned, compute wer for each utterance")
    args = parser.parse_args()
    run(args)