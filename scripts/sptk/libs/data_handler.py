#!/usr/bin/env python
# wujian@2018

import os
import sys
import glob
import pickle
import warnings

import _thread
import threading
import subprocess

from pathlib import Path

import librosa as audio_lib
import numpy as np
import scipy.io as sio

from io import TextIOWrapper, BytesIO
from . import kaldi_io as io
from .utils import forward_stft, read_wav, write_wav
from .scheduler import run_command

__all__ = [
    "ArchiveReader", "ArchiveWriter", "WaveWriter", "NumpyWriter",
    "SpectrogramReader", "ScriptReader", "WaveReader", "NumpyReader",
    "PickleReader", "MatReader", "BinaryReader"
]


def pipe_fopen(command, mode, background=True):
    if mode not in ["rb", "r"]:
        raise RuntimeError("Now only support input from pipe")

    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

    def background_command_waiter(command, p):
        p.wait()
        if p.returncode != 0:
            warnings.warn("Command \"{0}\" exited with status {1}".format(
                command, p.returncode))
            _thread.interrupt_main()

    if background:
        thread = threading.Thread(target=background_command_waiter,
                                  args=(command, p))
        # exits abnormally if main thread is terminated .
        thread.daemon = True
        thread.start()
    else:
        background_command_waiter(command, p)
    return p.stdout


def _fopen(fname, mode):
    """
    Extend file open function, to support 
        1) "-", which means stdin/stdout
        2) "$cmd |" which means pipe.stdout
    """
    if mode not in ["w", "r", "wb", "rb"]:
        raise ValueError("Unknown open mode: {mode}".format(mode=mode))
    if not fname:
        return None
    fname = fname.strip()
    if fname == "-":
        if mode in ["w", "wb"]:
            return sys.stdout.buffer if mode == "wb" else sys.stdout
        else:
            return sys.stdin.buffer if mode == "rb" else sys.stdin
    elif fname[-1] == "|":
        pin = pipe_fopen(fname[:-1], mode, background=(mode == "rb"))
        return pin if mode == "rb" else TextIOWrapper(pin)
    else:
        if mode in ["r", "rb"] and not os.path.exists(fname):
            raise FileNotFoundError(
                "Could not find common file: \"{}\"".format(fname))
        return open(fname, mode)


def _fclose(fname, fd):
    """
    Extend file close function, to support
        1) "-", which means stdin/stdout
        2) "$cmd |" which means pipe.stdout
        3) None type
    """
    if fname != "-" and fd and fname[-1] != "|":
        fd.close()


class ext_open(object):
    """
    To make _fopen/_fclose easy to use like:
    with open("egs.scp", "r") as f:
        ...
    """
    def __init__(self, fname, mode):
        self.fname = fname
        self.mode = mode

    def __enter__(self):
        self.fd = _fopen(self.fname, self.mode)
        return self.fd

    def __exit__(self, *args):
        _fclose(self.fname, self.fd)


def parse_scps(scp_path,
               value_processor=lambda x: x,
               num_tokens=2,
               restrict=True):
    """
    Parse kaldi's script(.scp) file with supported for stdin/pipe
    If num_tokens >= 2, function will check token number
    WARN: last line of scripts could not be None or with "\n" end
    """
    line = 0
    scp_dict = {}
    with ext_open(scp_path, "r") as f:
        for raw_line in f:
            scp_tokens = raw_line.strip().split()
            line += 1
            if scp_tokens[-1] == "|":
                key, value = scp_tokens[0], " ".join(scp_tokens[1:])
            else:
                if (num_tokens >= 2 and len(scp_tokens) != num_tokens) or (
                        restrict and len(scp_tokens) < 2):
                    raise RuntimeError(f"For {scp_path}, format error in " +
                                       f"line[{line:d}]: {raw_line}")
                if num_tokens == 2:
                    key, value = scp_tokens
                else:
                    key, value = scp_tokens[0], scp_tokens[1:]
            if key in scp_dict:
                raise ValueError(
                    f"Duplicated key \'{key}\' exists in {scp_path}")
            scp_dict[key] = value_processor(value)
    return scp_dict


class Reader(object):
    """
        Base class for sequential/random accessing, to be implemented
    """
    def __init__(self, scp_rspecifier, **kwargs):
        self.index_dict = parse_scps(scp_rspecifier, **kwargs)
        self.index_keys = list(self.index_dict.keys())

    def _load(self, key):
        # return path
        return self.index_dict[key]

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
        if type(index) not in [int, str]:
            raise IndexError(f"Unsupported index type: {type(index)}")
        if type(index) == int:
            # from int index to key
            num_utts = len(self.index_keys)
            if index >= num_utts or index < 0:
                raise KeyError("Interger index out of range, " +
                               f"{index:d} vs {num_utts:d}")
            index = self.index_keys[index]
        if index not in self.index_dict:
            raise KeyError(f"Missing utterance {index}!")
        return self._load(index)


class Writer(object):
    """
        Basic Writer class to be implemented
    """
    def __init__(self, obj_path_or_dir, scp_path=None, is_dir=False):
        self.scp_path = scp_path
        # if dump ark to output, then ignore scp
        if obj_path_or_dir == "-" and scp_path:
            warnings.warn("Ignore script output discriptor cause "
                          "dump archives to stdout")
            self.scp_path = None
        self.dump_out_dir = is_dir
        if is_dir:
            self.path_or_dir = Path(obj_path_or_dir).absolute()
            self.path_or_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.path_or_dir = os.path.abspath(obj_path_or_dir)

    def __enter__(self):
        # "wb" is important
        if not self.dump_out_dir:
            self.ark_file = _fopen(self.path_or_dir, "wb")
        self.scp_file = _fopen(self.scp_path, "w")
        return self

    def __exit__(self, *args):
        if not self.dump_out_dir:
            _fclose(self.path_or_dir, self.ark_file)
        _fclose(self.scp_path, self.scp_file)

    def write(self, key, data):
        raise NotImplementedError


class ArchiveReader(object):
    """
        Sequential Reader for Kalid's archive(.ark) object(support matrix/vector)
    """
    def __init__(self, ark_or_pipe, matrix=True):
        self.ark_or_pipe = ark_or_pipe
        self.matrix = matrix

    def __iter__(self):
        # to support stdin as input
        with ext_open(self.ark_or_pipe, "rb") as fd:
            for key, mat in io.read_ark(fd, matrix=self.matrix):
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
        or output from commands, egs:
            key1 sox /home/data/key1.wav -t wav - remix 1 |
    """
    def __init__(self, wav_scp, sample_rate=None, normalize=True):
        super(WaveReader, self).__init__(wav_scp)
        self.samp_rate = sample_rate
        self.normalize = normalize

    def read_internal(self, addr, beg=None, end=None):
        # return C x N or N
        sr, samps = read_wav(addr,
                             beg=beg,
                             end=end,
                             normalize=self.normalize,
                             return_rate=True)
        # if given samp_rate, check it
        if self.samp_rate is not None and sr != self.samp_rate:
            raise RuntimeError(
                f"SampleRate mismatch: {sr:d} vs {self.samp_rate:d}")
        return samps

    def read(self, key, beg=None, end=None):
        # return C x N matrix or N vector
        fname = self.index_dict[key]
        fname = fname.rstrip()
        # pipe open
        if fname[-1] == "|":
            stdout_shell, _ = run_command(fname[:-1], wait=True)
            return self.read_internal(BytesIO(stdout_shell))
        else:
            wav_list = glob.glob(fname)
            N = len(wav_list)
            if N == 0:
                raise RuntimeError("Could not find file matches " +
                                   f"template \'{fname}\'")
            elif N == 1:
                return self.read_internal(wav_list[0], beg=beg, end=end)
            else:
                # in sorted order, sentitive to beamforming
                return np.vstack([
                    self.read_internal(addr, beg=beg, end=end)
                    for addr in sorted(wav_list)
                ])

    def _load(self, key):
        return self.read(key)

    def maxabs(self, key):
        samps = self.read(key)
        return np.max(np.abs(samps))

    def duration(self, key):
        samps = self.read(key)
        return samps.shape[-1] / self.samp_rate

    def nsamps(self, key):
        samps = self.read(key)
        return samps.shape[-1]

    def power(self, key):
        samps = self.read(key)
        s = samps if samps.ndim == 1 else samps[0]
        return np.linalg.norm(s, 2)**2 / s.size


class SegmentWaveReader(Reader):
    """
    WaveReader with segments
    """
    def __init__(self, wav_scp, segments, sample_rate=None, normalize=True):
        def processor(x):
            wav, beg, end = x
            return {"wav": wav, "beg": float(beg), "end": float(end)}

        super(SegmentWaveReader, self).__init__(segments,
                                                num_tokens=4,
                                                value_processor=processor)
        self.wav_reader = WaveReader(wav_scp)

    def _load(self, key):
        info = self.index_dict[key]
        return self.wav_reader.read(info["wav"],
                                    beg=info["beg"],
                                    end=info["end"])


class NumpyReader(Reader):
    """
        Sequential/Random Reader for numpy's ndarray(*.npy) file
    """
    def __init__(self, npy_scp):
        super(NumpyReader, self).__init__(npy_scp)

    def _load(self, key):
        return np.load(self.index_dict[key])


class PickleReader(Reader):
    """
        Sequential/Random Reader for pickle object
    """
    def __init__(self, obj_scp):
        super(PickleReader, self).__init__(obj_scp)

    def _load(self, key):
        with open(self.index_dict[key], "rb") as f:
            obj = pickle.load(f)
        return obj


class MatReader(Reader):
    """
        Sequential/Random Reader for matlab matrix object
    """
    def __init__(self, mat_scp, key):
        super(MatReader, self).__init__(mat_scp)
        self.key = key

    def _load(self, key):
        mat_path = self.index_dict[key]
        mat_dict = sio.loadmat(mat_path)
        if self.key not in mat_dict:
            raise KeyError("Could not find \'{}\' in python "
                           "dictionary".format(self.key))
        return mat_dict[self.key]


class SpectrogramReader(WaveReader):
    """
        Sequential/Random Reader for single/multiple channel STFT
    """
    def __init__(self, wav_scp, normalize=True, **kwargs):
        super(SpectrogramReader, self).__init__(wav_scp, normalize=normalize)
        self.stft_kwargs = kwargs

    def _load(self, key):
        # get wave samples
        samps = super().read(key)
        if samps.ndim == 1:
            return forward_stft(samps, **self.stft_kwargs)
        else:
            N, _ = samps.shape
            # stft need input to be contiguous in memory
            # make samps.flags['C_CONTIGUOUS'] = True
            samps = np.ascontiguousarray(samps)
            return np.stack(
                [forward_stft(samps[c], **self.stft_kwargs) for c in range(N)])


class ScriptReader(Reader):
    """
        Reader for kaldi's scripts(for BaseFloat matrix)
    """
    def __init__(self, ark_scp, matrix=True):
        def addr_processor(addr):
            addr_token = addr.split(":")
            if len(addr_token) == 1:
                raise ValueError("Unsupported scripts address format")
            path, offset = ":".join(addr_token[0:-1]), int(addr_token[-1])
            return (path, offset)

        super(ScriptReader, self).__init__(ark_scp,
                                           value_processor=addr_processor)
        self.matrix = matrix
        self.fmgr = dict()

    def _open(self, obj, addr):
        if obj not in self.fmgr:
            self.fmgr[obj] = open(obj, "rb")
        arkf = self.fmgr[obj]
        arkf.seek(addr)
        return arkf

    def _load(self, key):
        path, addr = self.index_dict[key]
        fd = self._open(path, addr)
        io.expect_binary(fd)
        obj = io.read_general_mat(fd) if self.matrix else io.read_float_vec(fd)
        return obj


class BinaryReader(Reader):
    """
    Reader for binary objects(raw data)
    """
    def __init__(self, bin_scp, length=None, data_type="float32"):
        super(BinaryReader, self).__init__(bin_scp)
        supported_data = {
            "float32": np.float32,
            "float64": np.float64,
            "int32": np.int32,
            "int64": np.int64
        }
        if data_type not in supported_data:
            raise RuntimeError("Unsupported data type: {}".format(data_type))
        self.fmt = supported_data[data_type]
        self.length = length

    def _load(self, key):
        obj = np.fromfile(self.index_dict[key], dtype=self.fmt)
        if self.length is not None and obj.size != self.length:
            raise RuntimeError("Expect length {:d}, but got {:d}".format(
                self.length, obj.size))
        return obj


class ArchiveWriter(Writer):
    """
        Writer for kaldi's scripts && archive(for BaseFloat matrix)
    """
    def __init__(self, ark_path, scp_path=None, matrix=True, dtype=np.float32):
        if not ark_path:
            raise RuntimeError("Seem configure path of archives as None")
        super(ArchiveWriter, self).__init__(ark_path, scp_path)
        self.dtype = dtype
        self.dump_func = io.write_common_mat if matrix else io.write_float_vec

    def write(self, key, obj):
        if not isinstance(obj, np.ndarray):
            raise RuntimeError(
                f"Expect np.ndarray object, but got {type(obj)}")
        io.write_token(self.ark_file, key)
        # fix script generation bugs
        if self.path_or_dir != "-":
            offset = self.ark_file.tell()
        io.write_binary_symbol(self.ark_file)
        # cast to target type
        obj = obj.astype(self.dtype)
        self.dump_func(self.ark_file, obj)
        if self.scp_file:
            record = f"{key}\t{self.path_or_dir}:{offset:d}\n"
            self.scp_file.write(record)


class WaveWriter(Writer):
    """
        Writer for wave files
    """
    def __init__(self, dump_dir, scp_path=None, **wav_kwargs):
        super(WaveWriter, self).__init__(dump_dir, scp_path, is_dir=True)
        self.wav_kwargs = wav_kwargs

    def write(self, key, obj):
        if not isinstance(obj, np.ndarray):
            raise RuntimeError(
                f"Expect np.ndarray object, but got {type(obj)}")
        obj_path = self.path_or_dir / f"{key}.wav"
        write_wav(obj_path, obj, **self.wav_kwargs)
        if self.scp_file:
            self.scp_file.write(f"{key}\t{obj_path}\n")


class NumpyWriter(Writer):
    """
        Writer for numpy ndarray
    """
    def __init__(self, dump_dir, scp_path=None):
        super(NumpyWriter, self).__init__(dump_dir, scp_path, is_dir=True)

    def write(self, key, obj):
        if not isinstance(obj, np.ndarray):
            raise RuntimeError(
                f"Expect np.ndarray object, but got {type(obj)}")
        obj_path = self.path_or_dir / f"{key}.npy"
        np.save(obj_path, obj)
        if self.scp_file:
            self.scp_file.write(f"{key}\t{obj_path}\n")


class MatWriter(Writer):
    """
        Writer for Matlab's matrix
    """
    def __init__(self, dump_dir, scp_path=None):
        super(MatWriter, self).__init__(dump_dir, scp_path, is_dir=True)

    def write(self, key, obj):
        if not isinstance(obj, np.ndarray):
            raise RuntimeError(
                f"Expect np.ndarray object, but got {type(obj)}")
        obj_path = self.path_or_dir / f"{key}.npy"
        sio.savemat(obj_path, {"data": obj})
        if self.scp_file:
            self.scp_file.write(f"{key}\t{obj_path}\n")


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
