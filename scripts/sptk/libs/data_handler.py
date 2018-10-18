#!/usr/bin/env python
# wujian@2018

import os
import sys
import glob
import warnings
import librosa as audio_lib
import numpy as np

import libs.iobase as io
from libs.utils import stft, read_wav, parse_scps, get_logger

logger = get_logger(__name__)

__all__ = [
    "ArchiveReader",
    "ArchiveWriter",
    "SpectrogramReader",
    "ScriptReader",
    "WaveReader",
    "NumpyReader",
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
        if scp_path == "-":
            raise ValueError("Could not write .scp to stdout")
        self.scp_path = scp_path
        self.ark_path = ark_path
        if self.ark_path == "-" and self.scp_path:
            self.scp_path = None
            warnings.warn(
                "Ignore .scp output discriptor cause dump archives to stdout")

    def __enter__(self):
        self.ark_file = sys.stdout.buffer if self.ark_path == "-" else open(
            self.ark_path, "wb")
        # scp_path = "" or None
        self.scp_file = None if not self.scp_path else open(self.scp_path, "w")
        return self

    def __exit__(self, *args):
        if self.scp_file:
            self.scp_file.close()
        if self.ark_path != "-":
            self.ark_file.close()

    def write(self, key, data):
        raise NotImplementedError


class ArchiveReader(object):
    """
        Sequential Reader for Kalid's archive(.ark) object
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
    """
        Sequential/Random Reader for single/multiple channel wave
        Format of wav.scp follows Kaldi's definition:
            key1 /path/to/wav
            ...

        And /path/to/wav allowed to be a pattern, for example:
            key1 /home/data/key1.CH*.wav
        /home/data/key1.CH*.wav matches file /home/data/key1.CH{1,2,3..}.wav 
    """

    def __init__(self, wav_scp, sample_rate=None, normalize=True):
        super(WaveReader, self).__init__(wav_scp)
        self.samp_rate = sample_rate
        self.normalize = normalize

    def _query_flist(self, key):
        flist = glob.glob(self.index_dict[key])
        if not len(flist):
            raise RuntimeError(
                "Could not find file matches template \'{}\'".format(
                    self.index_dict[key]))
        return flist

    def _read(self, addr):
        # return C x N or N
        samp_rate, samps = read_wav(
            addr, normalize=self.normalize, return_rate=True)
        # if given samp_rate, check it
        if self.samp_rate is not None and samp_rate != self.samp_rate:
            raise RuntimeError("SampleRate mismatch: {:d} vs {:d}".format(
                samp_rate, self.samp_rate))
        return samps

    def _load(self, key):
        # return C x N matrix or N vector
        wav_list = self._query_flist(key)
        if len(wav_list) == 1:
            return self._read(wav_list[0])
        else:
            # in sorted order, sentitive to beamforming
            return np.vstack([self._read(addr) for addr in sorted(wav_list)])

    def samp_norm(self, key):
        samps = self._load(key)
        return np.max(np.abs(samps))


class NumpyReader(Reader):
    """
        Sequential/Random Reader for numpy's ndarray(*.npy) file
    """

    def __init__(self, npy_scp):
        super(NumpyReader, self).__init__(npy_scp)

    def _load(self, key):
        return np.load(self.index_dict[key])


class SpectrogramReader(Reader):
    """
        Sequential/Random Reader for single/multiple channel STFT
    """

    def __init__(self, wav_scp, normalize=True, **kwargs):
        super(SpectrogramReader, self).__init__(wav_scp)
        self.stft_kwargs = kwargs
        self.wave_reader = WaveReader(wav_scp, normalize=normalize)

    def _load(self, key):
        samps = self.wave_reader[key]
        if samps.ndim == 1:
            return stft(samps, **self.stft_kwargs)
        else:
            N, _ = samps.shape
            # stft need input to be contiguous in memory
            # make samps.flags['C_CONTIGUOUS'] = True
            samps = np.ascontiguousarray(samps)
            return np.stack(
                [stft(samps[c], **self.stft_kwargs) for c in range(N)])

    def samp_norm(self, key):
        return self.wave_reader.samp_norm(key)


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
        io.write_binary_symbol(self.ark_file)
        io.write_common_mat(self.ark_file, matrix)
        abs_path = os.path.abspath(self.ark_path)
        if self.scp_file:
            offset = self.ark_file.tell()
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
