#!/usr/bin/env python
# wujian@2018

import os
import sys
import glob
import warnings
import librosa as audio_lib
import numpy as np

import libs.iobase as io
from libs.utils import stft, parse_scps, get_logger

logger = get_logger(__name__)

__all__ = [
    "ArchiveReader", "ArchiveWriter", "SpectrogramReader", "ScriptReader", "WaveReader"
]

class Reader(object):
    """
        Base class for sequential/random accessing, to be implemented
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
                raise KeyError("Interger index out of range, {} vs {}".format(
                    index, num_utts))
            key = self.index_keys[index]
            return self._load(key)
        elif type(index) is str:
            if index not in self.index_dict:
                raise KeyError("Missing utterance {}!".format(index))
            return self._load(index)
        else:
            raise IndexError("Unsupported index type: {}".format(type(index)))


class Writer(object):
    """
        Base Writer class to be implemented
    """

    def __init__(self, ark_path, scp_path=None):
        if scp_path == '-':
            raise ValueError("Could not write .scp to stdout")
        self.scp_path = scp_path
        self.ark_path = ark_path
        if self.ark_path == '-' and self.scp_path:
            self.scp_path = None
            warnings.warn("Ignore .scp output discriptor")

    def __enter__(self):
        self.scp_file = None if self.scp_path is None else open(
            self.scp_path, "w")
        self.ark_file = sys.stdout if self.ark_path == '-' else open(
            self.ark_path, "wb")
        return self

    def __exit__(self, *args):
        if self.scp_file:
            self.scp_file.close()
        if self.ark_path != '-':
            self.ark_file.close()

    def write(self, key, data):
        raise NotImplementedError

class ArchiveReader(object):
    """
        Sequential Reader for .ark object
    """
    def __init__(self, ark_path):
        if not os.path.exists(ark_path):
            raise FileNotFoundError("Could not find {}".format(ark_path))
        self.ark_path = ark_path
    
    def __iter__(self):
        with open(self.ark_path, "rb") as fd:
            for key, mat in io.read_ark(fd):
                yield key, mat


class WaveReader(Reader):
    def __init__(self, scp_path, sample_rate=None):
        super(WaveReader, self).__init__(scp_path)
        self.sample_rate = sample_rate

    def _load(self, key):
        wav_addr = self.index_dict[key]
        samps, _ = audio_lib.load(wav_addr, sr=self.sample_rate)
        return samps

class NumpyReader(Reader):
    def __init__(self, scp_path):
        super(NumpyReader, self).__init__(scp_path)

    def _load(self, key):
        return np.load(self.index_dict[key])

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

    def _query_flist(self, key):
        flist = glob.glob(self.index_dict[key])
        if not len(flist):
            raise RuntimeError(
                "Could not find file matches template \'{}\'".format(
                    self.index_dict[key]))
        return flist

    # stft, single or multi-channal
    def _load(self, key):
        flist = self._query_flist(key)
        if len(flist) == 1:
            return stft(flist[0], **self.stft_kwargs)
        else:
            return np.array(
                [stft(f, **self.stft_kwargs) for f in sorted(flist)])

    def samp_norm(self, key):
        flist = self._query_flist(key)
        if len(flist) == 1:
            samps = audio_lib.load(flist[0], sr=None)[0]
            return np.linalg.norm(samps, np.inf)
        else:
            samps_list = [audio_lib.load(f, sr=None)[0] for f in flist]
            return sum([np.linalg.norm(samps, np.inf)
                        for samps in samps_list]) / len(flist)

class ScriptReader(Reader):
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

        super(ScriptReader, self).__init__(
            ark_scp, addr_processor=addr_processor)

    def _load(self, key):
        path, offset = self.index_dict[key]
        with open(path, 'rb') as f:
            f.seek(offset)
            io.expect_binary(f)
            ark = io.read_general_mat(f)
        return ark


class ArchiveWriter(Writer):
    """
        Writer for kaldi's scripts && archive(for BaseFloat matrix)
    """

    def __init__(self, ark_path, scp_path=None):
        super(ArchiveWriter, self).__init__(ark_path, scp_path)

    def write(self, key, matrix):
        io.write_token(self.ark_file, key)
        offset = self.ark_file.tell()
        io.write_binary_symbol(self.ark_file)
        io.write_common_mat(self.ark_file, matrix)
        abs_path = os.path.abspath(self.ark_path)
        if self.scp_file:
            self.scp_file.write("{}\t{}:{:d}\n".format(key, abs_path, offset))


def test_archive_writer(ark, scp):
    with ArchiveWriter(ark, scp) as writer:
        for i in range(10):
            mat = np.random.rand(100, 20)
            writer.write("mat-{:d}".format(i), mat)
    print("TEST *test_archive_writer* DONE!")


def test_script_reader(egs):
    scp_reader = ScriptReader(egs)
    for key, mat in scp_reader:
        print("{}: {}".format(key, mat.shape))
    print("TEST *test_script_reader* DONE!")


if __name__ == "__main__":
    test_archive_writer("egs.ark", "egs.scp")
    test_script_reader("egs.scp")
