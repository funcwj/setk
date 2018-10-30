"""
Some customized action for argparse
"""
import argparse


def str_to_float_tuple(string, sep=","):
    tokens = string.split(sep)
    if len(tokens) == 1:
        raise ValueError("Get only one token by sep={0}, string={1}".format(
            sep, string))
    floats = map(float, tokens)
    return tuple(floats)


def str_to_bool(value):
    if value == "true":
        return True
    elif value == "false":
        return False
    else:
        raise ValueError


class StrToBoolAction(argparse.Action):
    """
    Since argparse.store_true is not very convenient
    """

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            setattr(namespace, self.dest, str_to_bool(values))
        except ValueError:
            raise Exception("Unknown value {0} for --{1}".format(
                values, self.dest))


class StrToFloatTupleAction(argparse.Action):
    """
    Egs: 2,10 -> (2, 10)
    """

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            setattr(namespace, self.dest, str_to_float_tuple(values))
        except ValueError:
            raise Exception("Unknown value {0} for --{1}".format(
                values, self.dest))


def get_stft_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--frame-length",
        type=int,
        default=1024,
        dest="frame_length",
        help="Frame length in number of samples(related to sample frequency)")
    parser.add_argument(
        "--frame-shift",
        type=int,
        default=256,
        dest="frame_shift",
        help="Frame shift in number of samples(related to sample frequency)")
    parser.add_argument(
        "--center",
        action=StrToBoolAction,
        default=True,
        dest="center",
        help="Value of parameter \'center\' in librosa.stft functions")
    parser.add_argument(
        "--window",
        default="hann",
        dest="window",
        help="Type of window function, see scipy.signal.get_window")
    return parser
