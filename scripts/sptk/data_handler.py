#!/usr/bin/env python
# wujian@2018

import os
import glob
import random
import warnings
import librosa as audio_lib
import numpy as np
import iobase as io

from utils import stft, parse_scps, get_logger

logger = get_logger(__name__)


class Reader(object):
    """
        Base class, to be implemented
    """

    def __init__(self, scp_path, addr_processor=lambda x: x):
        if not os.path.exists(scp_path):
            raise FileNotFoundError("Could not find file {}".format(scp_path))
        self.index_dict = parse_scps(scp_path, addr_processor=addr_processor)
        self.index_keys = [key for key in self.index_dict.keys()]

    def _load(self, key):
        raise NotImplementedError

    # number of utterance
    def __len__(self):
        return len(self.index_dict)

    # avoid key error
    def __contains__(self, key):
        return key in self.index_dict

    # sequential index
    def __iter__(self):
        for key in self.index_keys:
            yield key, self._load(key)

    # random index, support str/int as index
    def __getitem__(self, index):
        if type(index) == int:
            num_utts = len(self.index_keys)
            if index >= num_utts or index < 0:
                raise KeyError("Interger index out of range, {} vs {}".format(index, num_utts))
            key = self.index_keys[index]
            return self._load(key)
        elif type(index) is str:
            if index not in self.index_dict:
                raise KeyError("Missing utterance {}!".format(index))
            return self._load(index)
        else:
            raise IndexError("Unsupported index type: {}".format(type(index)))


class WaveReader(Reader):
    def __init__(self, scp_path, sample_rate=None):
        super(WaveReader, self).__init__(scp_path)
        self.sample_rate = sample_rate

    def _load(self, key):
        wav_addr = self.index_dict[key]
        samps, _ = audio_lib.load(wav_addr, sr=self.sample_rate)
        return samps


class SpectrogramReader(Reader):
    """
        Wrapper for short-time fourier transform of wave scripts
    """

    def __init__(self, wave_scp, **kwargs):
        super(SpectrogramReader, self).__init__(wave_scp)
        if "return_samps" in kwargs and kwargs["return_samps"]:
            warnings.warn("Argument --return_samps is True here, ignore it")
            kwargs["return_samps"] = False
        self.stft_kwargs = kwargs

    # stft, single or multi-channal
    def _load(self, key):
        flist = glob.glob(self.index_dict[key])
        if not len(flist):
            raise RuntimeError(
                "Could not find file matches template \'{}\'".format(
                    self.index_dict[key]))
        if len(flist) == 1:
            return stft(flist[0], **self.stft_kwargs)
        else:
            return np.array(
                [stft(f, **self.stft_kwargs) for f in sorted(flist)])


class ArchieveReader(Reader):
    """
        Reader for kaldi's scripts(for BaseFloat matrix)
    """

    def __init__(self, ark_scp):
        def addr_processor(addr):
            addr_token = addr.split(":")
            if len(addr_token) == 1:
                raise ValueError("Unsupported scripts address format")
            path, offset = ":".join(addr_token[0:-1]), int(addr_token[-1])
            return (path, offset)

        super(ArchieveReader, self).__init__(
            ark_scp, addr_processor=addr_processor)

    def _load(self, key):
        path, offset = self.index_dict[key]
        with open(path, 'rb') as f:
            f.seek(offset)
            io.expect_binary(f)
            ark = io.read_general_mat(f)
        return ark


class ArchieveWriter(object):
    """
        Writer for kaldi's scripts && archieve(for BaseFloat matrix)
    """

    def __init__(self, ark_path, scp_path=None):
        self.scp_path = scp_path
        self.ark_path = ark_path

    def __enter__(self):
        self.scp_file = None if self.scp_path is None else open(
            self.scp_path, "w")
        self.ark_file = open(self.ark_path, "wb")
        return self

    def __exit__(self, type, value, trace):
        if self.scp_file:
            self.scp_file.close()
        self.ark_file.close()

    def write(self, key, matrix):
        io.write_token(self.ark_file, key)
        offset = self.ark_file.tell()
        # binary symbol
        io.write_binary_symbol(self.ark_file)
        io.write_common_mat(self.ark_file, matrix)
        if self.scp_file:
            self.scp_file.write("{}\t{}:{:d}\n".format(
                key, os.path.abspath(self.ark_path), offset))


def test_archieve_writer(ark, scp):
    with ArchieveWriter(ark, scp) as writer:
        for i in range(10):
            mat = np.random.rand(100, 20)
            writer.write("mat-{:d}".format(i), mat)
    print("TEST *test_archieve_writer* DONE!")


def test_archieve_reader(egs):
    ark_reader = ArchieveReader(egs)
    for key, mat in ark_reader:
        print("{}: {}".format(key, mat.shape))
    print("TEST *test_archieve_reader* DONE!")


if __name__ == "__main__":
    test_archieve_writer("egs.ark", "egs.scp")
    test_archieve_reader("egs.scp")