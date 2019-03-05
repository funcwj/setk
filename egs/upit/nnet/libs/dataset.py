# wujian@2019

import torch as th
import numpy as np

from queue import PriorityQueue

from torch.nn.utils.rnn import pad_sequence
from kaldi_python_io import ArchiveReader, ScriptReader, Reader


def make_pitloader(feats_scp,
                   feats_kwargs,
                   loader_kwargs,
                   batch_size=8,
                   cache_size=32):
    processor = Processor(feats_scp, **feats_kwargs)
    perutt_loader = PITLoader(processor, **loader_kwargs)
    return UttLoader(
        perutt_loader, batch_size=batch_size, cache_size=cache_size)


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


class UttLoader(object):
    """
    Dataloader for batch of utterances (online loading for memory effcient)
    """

    def __init__(self, perutt_loader, batch_size=16, cache_size=32):
        self.max_size = cache_size * batch_size
        self.num_utts = batch_size
        self.pqueue = PriorityQueue()
        self.loader = perutt_loader

    def collate(self, obj_list):
        """
        Simple collate function
        """
        peek = obj_list[0]
        if isinstance(peek, np.ndarray):
            return self.collate(
                [th.tensor(arr, dtype=th.float32) for arr in obj_list])
        elif isinstance(peek, th.Tensor):
            return th.stack(obj_list) if peek.dim() == 1 else pad_sequence(
                obj_list, batch_first=True)
        elif isinstance(peek, float):
            return th.tensor(obj_list, dtype=th.float32)
        elif isinstance(peek, int):
            return th.tensor(obj_list, dtype=th.int64)
        elif isinstance(peek, list):
            return [self.collate(nth_seq) for nth_seq in zip(*obj_list)]
        elif isinstance(peek, dict):
            return {
                key: self.collate([d[key] for d in obj_list])
                for key in peek
            }
        else:
            raise RuntimeError("Unknown object type: {}".format(type(peek)))

    def make_egs(self, eg_list):
        """
        Make one egs from a list of nnet example
        """
        egs = self.collate(eg_list)
        egs["xlen"] = self.collate([len(eg) for eg in eg_list])
        return egs

    def fetch_batch(self, drop_last=False):
        while True:
            try:
                eg = next(self.load_iter)
                self.pqueue.put(eg)
            except StopIteration:
                # set stop
                self.stop_iter = True
                break
            if self.pqueue.qsize() == self.max_size:
                break
        # pop sorted result
        egs_sorted = []
        while self.pqueue.qsize():
            eg = self.pqueue.get()
            egs_sorted.append(eg)
        N = len(egs_sorted)
        cur = 0
        # make egs for #num_utts
        egs_batch = []
        while cur < N:
            if drop_last and cur + self.num_utts > N:
                break
            end = min(N, cur + self.num_utts)
            egs = self.make_egs(egs_sorted[cur:end])
            cur = end
            egs_batch.append(egs)
        return egs_batch

    def __iter__(self):
        self.load_iter = iter(self.loader)
        # reset flags
        self.stop_iter = False
        while True:
            if self.stop_iter:
                break
            obj_list = self.fetch_batch()
            for obj in obj_list:
                yield obj