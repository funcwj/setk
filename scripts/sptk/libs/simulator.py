# wujian@2018

import random
import argparse

from libs.data_handler import parse_scps

class DictSampler(object):
    def __init__(self, scp):
        self.dict = parse_scps(scp)
        self.keys = [key for key in self.dict]

    def __len__(self):
        return len(self.dict)

    def sample(self, num_items):
        keys = random.sample(self.keys, num_items)
        vals = [self.dict[key] for key in keys]
        return vals[0] if num_items == 1 else vals


def str_to_float_tuple(string, sep=","):
    tokens = string.split(sep)
    if len(tokens) == 1:
        raise ValueError("Get only one token by sep={0}, string={1}".format(
            sep, string))
    floats = map(float, tokens)
    return tuple(floats)


class StrToFloatTupleAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            setattr(namespace, self.dest, str_to_float_tuple(values))
        except ValueError:
            raise Exception("Unknown value {0} for --{1}".format(
                values, self.dest))
