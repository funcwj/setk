# wujian@2018

import os
import json
import errno
import logging


def get_logger(
        name,
        format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
        date_format="%Y-%m-%d %H:%M:%S",
        file=False):
    """
    Get logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # file or console
    handler = logging.StreamHandler() if not file else logging.FileHandler(
        name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def dump_json(obj, fdir, name):
    """
    Serialize python object as json
    """
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    with open(os.path.join(fdir, name), "w") as f:
        json.dump(obj, f, indent=4, sort_keys=False)


def load_json(fdir, name):
    """
    Load json file as python object
    """
    path = os.path.join(fdir, name)
    if not os.path.exists(path):
        raise FileNotFoundError("Could not find json file: {}".format(path))
    with open(path, "r") as f:
        obj = json.load(f)
    return obj


def make_dir(fdir):
    """
    Make directory 
    """
    if not fdir or os.path.exists(fdir):
        return
    try:
        os.makedirs(fdir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise RuntimeError("Error exists when mkdir -p {}".format(fdir))