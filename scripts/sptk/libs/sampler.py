# wujian@2018

import os
import random
import argparse

from .data_handler import Reader, WaveReader, NumpyReader, ScriptReader
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
    Sampler class for scripts
    """
    def __init__(self, scp, utt2dur):
        self.val = Reader(scp, num_tokens=-1, restrict=False)
        self.dur = Reader(utt2dur, value_processor=lambda x: float(x))

    def __len__(self):
        return len(self.val)

    def __getitem__(self, key):
        return self.val[key], self.dur[key]

    def sample(self, num_items):
        keys = random.sample(self.val.index_keys, num_items)
        egs = [{"loc": self.val[key], "dur": self.dur[key]} for key in keys]
        return egs[0] if num_items == 1 else egs