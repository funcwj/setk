#!/usr/bin/env python

# wujian@2018

import numpy as np
from kaldi_python_io import ArchiveReader, ScriptReader, Reader


class Processor(object):
    def __init__(self,
                 feats_scp,
                 apply_log=True,
                 norm_means=True,
                 norm_vars=True,
                 cmvn=None,
                 shuf=False,
                 lctx=0,
                 rctx=0):
        # shuffle or not
        scp_src = "shuf {0} |".format(feats_scp) if shuf else feats_scp
        # log or linear
        pipe_cmd = "copy-matrix {opts} \"scp:{scp}\" ark:- | ".format(
            opts="--apply-log" if apply_log else "", scp=scp_src)
        # cmvn opts
        cmvn_cmd = ""
        if norm_means or norm_vars:
            cmvn_cmd = "apply-cmvn-perutt {0} --norm-means={1} " \
                "--norm-vars={2} ark:- ark:- | ".format(
                "--gcmvn={}".format(cmvn) if cmvn else "",
                "true" if norm_means else "false",
                "true" if norm_vars else "false")
        pipe_cmd += cmvn_cmd
        # splice opts
        splice_cmd = ""
        if lctx + rctx != 0:
            splice_cmd = "splice-feats --left-context={:d} " \
                "--right-context={:d} ark:- ark:- |".format(
                lctx, rctx)
        pipe_cmd += splice_cmd

        self.cmd = pipe_cmd
        self.reader = ArchiveReader(pipe_cmd)

    def __iter__(self):
        for key, feats in self.reader:
            yield key, feats


class UttExample(dict):
    """
    A utterance example used for neural network training
    """

    def __init__(self, priority="feats"):
        super(UttExample, self).__init__()
        self.priority = priority

    def __len__(self):
        return self[self.priority].shape[0]

    def __lt__(self, other):
        return len(self) > len(other)


class PITLoader(object):
    """
    A spectial PeruttLoader designed for PIT
    """

    def __init__(self, processor, linear_x="", spatial="", linear_y=None):
        self.processor = processor
        self.linear_x = ScriptReader(linear_x)
        self.linear_y = [ScriptReader(ly) for ly in linear_y]
        self.spatial = ScriptReader(spatial) if spatial else None

    def _make_egs(self, uttid):
        eg = UttExample()
        for reader in self.linear_y:
            if uttid not in reader:
                return None
        if uttid in self.linear_x:
            eg["lx"] = self.linear_x[uttid]
            eg["ly"] = [reader[uttid] for reader in self.linear_y]
            return eg
        return None

    def __iter__(self):
        for uttid, feats in self.processor:
            eg = self._make_egs(uttid)
            if eg is not None:
                if self.spatial:
                    spatial = self.spatial[uttid]
                    feats = np.hstack([feats, spatial])
                eg["feats"] = feats
                yield eg
