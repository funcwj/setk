#!/usr/bin/env python
# wujian@2019

import argparse

import numpy as np

from collections import defaultdict

from libs.data_handler import WaveReader, Reader
from mir_eval.separation import bss_eval_sources


class AudioReader(object):
    def __init__(self, spks_scp):
        self.wav_reader = [WaveReader(scp) for scp in spks_scp.split(",")]

    def __len__(self):
        return len(self.wav_reader[0])

    def _index(self, key):
        data = [reader[key] for reader in self.wav_reader]
        return np.stack(data, axis=0)

    def __getitem__(self, key):
        return self._index(key)

    def __iter__(self):
        ref = self.wav_reader[0]
        for key in ref.index_keys:
            yield key, self._index(key)


class Report(object):
    def __init__(self, spk2class=None):
        self.s2c = Reader(spk2class) if spk2class else None
        self.snr = defaultdict(float)
        self.cnt = defaultdict(int)

    def add(self, key, val):
        cls_str = "NG"
        if self.s2c:
            cls_str = self.s2c[key]
        self.snr[cls_str] += val
        self.cnt[cls_str] += 1

    def report(self):
        print("SDR(dB) Report: ")
        tot_utt = sum([self.cnt[cls_str] for cls_str in self.cnt])
        tot_snr = sum([self.snr[cls_str] for cls_str in self.snr])
        print("Total: {:d}/{:.3f}".format(tot_utt, tot_snr / tot_utt))
        for cls_str in self.snr:
            cls_snr = self.snr[cls_str]
            num_utt = self.cnt[cls_str]
            print("\t{}: {:d}/{:.3f}".format(cls_str, num_utt,
                                             cls_snr / num_utt))


def run(args):

    sep_reader = AudioReader(args.sep_scp)
    ref_reader = AudioReader(args.ref_scp)
    each_utt = open(args.per_utt, "w") if args.each_utt else None
    reporter = Report(args.spk2class)
    # sep: N x S
    for key, sep in sep_reader:
        # ref: N x S
        ref = ref_reader[key]
        # keep same shape
        nsamps = min(sep.shape[-1], ref.shape[-1])
        sdr, _, _, _ = bss_eval_sources(ref[:, :nsamps], sep[:, :nsamps])
        sdr = np.mean(sdr)
        reporter.add(key, sdr)
        if each_utt:
            each_utt.write("{}\t{:.2f}\n".format(key, sdr))

    reporter.report()
    if each_utt:
        each_utt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to eval speech separation (SDR) using "
        "mir_eval (https://github.com/craffel/mir_eval)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("sep_scp",
                        type=str,
                        help="Separated speech scripts, waiting for measure"
                        "(support multi-speaker, egs: spk1.scp,spk2.scp)")
    parser.add_argument("ref_scp",
                        type=str,
                        help="Reference speech scripts, as ground truth for "
                        "separation evaluation")
    parser.add_argument("--spk2class",
                        type=str,
                        default="",
                        help="If assigned, report results"
                        " per class (gender or degree)")
    parser.add_argument("--per-utt",
                        type=str,
                        default="",
                        help="If assigned, report snr "
                        "improvement for each utterance")
    args = parser.parse_args()
    run(args)