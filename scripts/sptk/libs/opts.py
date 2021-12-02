# wujian@2018
"""
Some customized action for argparse
"""
import argparse
from distutils.util import strtobool


def str2tuple(string, sep=","):
    """
    Map "1.0,2,0" => (1.0, 2.0)
    """
    tokens = string.split(sep)
    # if len(tokens) == 1:
    #     raise ValueError("Get only one token by " +
    #                      f"sep={sep}, string={string}")
    floats = map(float, tokens)
    return tuple(floats)


class StftParser(object):
    """
    STFT argparser
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--frame-len",
                        type=int,
                        default=512,
                        help="Frame length in number of samples "
                        "(related to sample frequency)")
    parser.add_argument("--frame-hop",
                        type=int,
                        default=256,
                        help="Frame shift in number of samples "
                        "(related to sample frequency)")
    parser.add_argument("--center",
                        type=strtobool,
                        default=True,
                        help="Value of parameter \'center\' in "
                        "librosa.stft functions")
    parser.add_argument("--round-power-of-two",
                        type=strtobool,
                        default=True,
                        help="If true, pad fft size to power of two")
    parser.add_argument("--window",
                        type=str,
                        default="hann",
                        help="Type of window function, "
                        "see scipy.signal.get_window")
