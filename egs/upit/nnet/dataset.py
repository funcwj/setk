# wujian@2019

import torch as th
import numpy as np

from queue import PriorityQueue
from torch.nn.utils.rnn import pad_sequence

from loader import Processor, PITLoader


def make_pitloader(feats_scp,
                   feats_kwargs,
                   loader_kwargs,
                   num_utts=8,
                   cache_size=32):
    processor = Processor(feats_scp, **feats_kwargs)
    perutt_loader = PITLoader(processor, **loader_kwargs)
    return UttLoader(perutt_loader, num_utts=num_utts, cache_size=cache_size)


class UttLoader(object):
    """
    Dataloader for batch of utterances
    """

    def __init__(self, perutt_loader, num_utts=16, cache_size=32):
        self.max_size = cache_size * num_utts
        self.num_utts = num_utts
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