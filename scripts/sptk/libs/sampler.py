# wujian@2020

import os
import random
import argparse

from .data_handler import Reader
from .opts import str2tuple


class UniformSampler(object):
    """
    A uniform sampler class
    """
    def __init__(self, tuple_or_str):
        if isinstance(tuple_or_str, (list, tuple)):
            self.min, self.max = tuple_or_str
        else:
            self.min, self.max = str2tuple(tuple_or_str)

    def sample(self):
        return random.uniform(self.min, self.max)


class ScriptSampler(object):
    """
    Sampler class for audio location & duration pair
    """
    def __init__(self, scp, utt2dur, dur=""):
        if not utt2dur:
            raise RuntimeError("utt2dur is None")
        self.location = Reader(scp, num_tokens=2, restrict=True)
        self.duration = Reader(utt2dur, value_processor=lambda x: float(x))
        self.index_keys = self.location.index_keys
        if dur:
            min_dur, max_dur = str2tuple(dur)
            filter_keys = []
            for key, dur in self.duration:
                if dur >= min_dur and dur <= max_dur:
                    filter_keys.append(key)
            self.index_keys = filter_keys

    def __len__(self):
        return len(self.index_keys)

    def __getitem__(self, key):
        return {"loc": self.location[key], "dur": self.duration[key]}

    def sample(self, num_items):
        keys = random.sample(self.index_keys, num_items)
        egs = [{
            "loc": self.location[key],
            "dur": self.duration[key]
        } for key in keys]
        return egs[0] if num_items == 1 else egs