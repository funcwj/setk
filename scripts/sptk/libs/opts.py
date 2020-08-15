# wujian@2018
"""
Some customized action for argparse
"""
import argparse


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


def str2bool(value):
    """
    Map "true"/"false" => True/False
    """
    if value.lower() in ["true", "t", "1", "yes", "y"]:
        return True
    elif value.lower() in ["false", "f", "no", "n", "0"]:
        return False
    else:
        raise ValueError


class StrToBoolAction(argparse.Action):
    """
    Since argparse.store_true is not very convenient
    """
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            setattr(namespace, self.dest, str2bool(values))
        except ValueError:
            raise Exception(f"Unknown value {values} for --{self.dest}")


class StrToFloatTupleAction(argparse.Action):
    """
    Egs: 2,10 -> (2, 10)
    """
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            setattr(namespace, self.dest, str2tuple(values))
        except ValueError:
            raise Exception(f"Unknown value {values} for --{self.dest}")


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
                        action=StrToBoolAction,
                        default=True,
                        help="Value of parameter \'center\' in "
                        "librosa.stft functions")
    parser.add_argument("--round-power-of-two",
                        action=StrToBoolAction,
                        default=True,
                        help="If true, pad fft size to power of two")
    parser.add_argument("--window",
                        type=str,
                        default="hann",
                        help="Type of window function, "
                        "see scipy.signal.get_window")
