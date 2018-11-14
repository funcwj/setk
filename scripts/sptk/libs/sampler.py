# wujian@2018

import os
import random
import argparse

from .data_handler import Reader, WaveReader, NumpyReader, ScriptReader


class Sampler(object):
    def __init__(self, obj_reader):
        if not isinstance(obj_reader, Reader):
            raise TypeError(
                "obj_reader is not a instance of libs.data_handler.Reader")
        self.reader = obj_reader

    def __len__(self):
        return len(self.reader)

    def sample(self, num_items):
        keys = random.sample(self.reader.index_keys, num_items)
        vals = [self.reader[key] for key in keys]
        return vals[0] if num_items == 1 else vals


class ScriptSampler(Sampler):
    def __init__(self, raw_scp, **kwargs):
        super(ScriptSampler, self).__init__(Reader(raw_scp, **kwargs))


class WaveSampler(Sampler):
    def __init__(self, wav_scp, **kwargs):
        super(WaveSampler, self).__init__(WaveReader(wav_scp, **kwargs))


class NumpySampler(Sampler):
    def __init__(self, npy_scp, **kwargs):
        super(NumpySampler, self).__init__(NumpyReader(npy_scp, **kwargs))


class ArchiveSampler(Sampler):
    def __init__(self, ark_scp, **kwargs):
        super(ArchiveSampler, self).__init__(ScriptReader(ark_scp, **kwargs))


# ...