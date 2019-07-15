#!/usr/bin/env python

# wujian@2018
"""
Compute Si-SDR as the evaluation metric
"""

import argparse

from tqdm import tqdm

from collections import defaultdict
from libs.metric import si_snr, permute_si_snr
from libs.data_handler import WaveReader, Reader


class SpeakersReader(object):
    def __init__(self, scps):
        split_scps = scps.split(",")
        if len(split_scps) == 1:
            raise RuntimeError("Construct SpeakersReader need more "
                               "than one script, got {}".format(scps))
        self.readers = [WaveReader(scp) for scp in split_scps]

    def __len__(self):
        first_reader = self.readers[0]
        return len(first_reader)

    def __getitem__(self, key):
        data = []
        for reader in self.readers:
            wave = reader[key]
            # if multi-channel, choose ch0 as reference
            # we always use ch0 as training targets
            data.append(wave if wave.ndim == 1 else wave[0])
        return data

    def __iter__(self):
        first_reader = self.readers[0]
        for key in first_reader.index_keys:
            yield key, self[key]


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
        print("Si-SDR(dB) Report: ")
        tot_utt = sum([self.cnt[cls_str] for cls_str in self.cnt])
        tot_snr = sum([self.snr[cls_str] for cls_str in self.snr])
        print("Total: {:d}/{:.3f}".format(tot_utt, tot_snr / tot_utt))
        for cls_str in self.snr:
            cls_snr = self.snr[cls_str]
            num_utt = self.cnt[cls_str]
            print("\t{}: {:d}/{:.3f}".format(cls_str, num_utt,
                                             cls_snr / num_utt))


def run(args):
    single_speaker = len(args.sep_scp.split(",")) == 1
    reporter = Report(args.spk2class)
    utt_snr = open(args.per_utt, "w") if args.per_utt else None
    utt_ali = open(args.utt_ali, "w") if args.utt_ali else None

    if single_speaker:
        sep_reader = WaveReader(args.sep_scp)
        ref_reader = WaveReader(args.ref_scp)
        for key, sep in tqdm(sep_reader):
            ref = ref_reader[key]
            if sep.size != ref.size:
                end = min(sep.size, ref.size)
                sep = sep[:end]
                ref = ref[:end]
            snr = si_snr(sep, ref)
            reporter.add(key, snr)
            if utt_snr:
                utt_snr.write("{}\t{:.2f}\n".format(key, snr))
    else:
        sep_reader = SpeakersReader(args.sep_scp)
        ref_reader = SpeakersReader(args.ref_scp)
        for key, sep_list in tqdm(sep_reader):
            ref_list = ref_reader[key]
            if sep_list[0].size != ref_list[0].size:
                end = min(sep_list[0].size, ref_list[0].size)
                sep_list = [s[:end] for s in sep_list]
                ref_list = [s[:end] for s in ref_list]
            snr, ali = permute_si_snr(sep_list, ref_list, align=True)
            reporter.add(key, snr)
            if utt_snr:
                utt_snr.write(f"{key}\t{snr:.2f}\n")
            if utt_ali:
                ali_str = " ".join(map(str, ali))
                utt_ali.write(f"{key}\t{ali_str}\n")
    reporter.report()
    if utt_snr:
        utt_snr.close()
    if utt_ali:
        utt_ali.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to compute SI-SDR, as metric of the separation quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("sep_scp",
                        type=str,
                        help="Separated speech scripts, waiting for measure"
                        "(support multi-speaker, egs: spk1.scp,spk2.scp)")
    parser.add_argument("ref_scp",
                        type=str,
                        help="Reference speech scripts, as ground truth for"
                        " Si-SDR computation")
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
    parser.add_argument("--utt-ali",
                        type=str,
                        default="",
                        help="If assigned, output audio alignments")
    args = parser.parse_args()
    run(args)