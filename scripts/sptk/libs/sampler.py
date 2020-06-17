# wujian@2020

import os
import random
import argparse

from .data_handler import ScpReader
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
