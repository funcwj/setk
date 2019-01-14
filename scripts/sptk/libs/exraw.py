#!/usr/bin/env python

# wujian@2019
"""
Provide a dependent io module for ndarray
"""

import struct

import numpy as np


def _serialize(fd, arr):
    support_dtype = {
        "float32": 'f',
        "float64": 'd',
        "int32": 'i',
        "int64": 'q'
    }
    dtype = str(arr.dtype)
    if dtype not in support_dtype:
        raise TypeError("Unsupported dtype: {}".format(arr.dtype))
    dim = arr.ndim
    if dim >= 3:
        raise RuntimeError("Accept only 1/2D ndarray, got {:d}".format(dim))
    # dtype flag
    fd.write(str.encode(support_dtype[dtype]))
    if dim == 1:
        fd.write(str.encode("v"))
        fd.write(struct.pack('i', arr.size))
    else:
        fd.write(str.encode("m"))
        M, N = arr.shape
        fd.write(struct.pack('i', M))
        fd.write(struct.pack('i', N))
    raw_data = arr.tobytes()
    fd.write(raw_data)


def _deserialize(fd, addr=None):
    support_dtype = {
        'f': (np.float32, 4),
        'd': (np.float64, 8),
        'i': (np.int32, 4),
        'q': (np.int64, 8),
    }
    if addr:
        fd.seek(addr)
    # numpy dtype
    dtype = bytes.decode(fd.read(1))
    if dtype not in support_dtype:
        raise TypeError(
            "Unsupported dtype string, expect f/d/i/q, got {}".format(dtype))
    # array type
    atype = bytes.decode(fd.read(1))
    if atype not in ["v", "m"]:
        raise TypeError(
            "Unsupported atype string: expect v/m, got {}".format(atype))
    np_dtype, sizeof = support_dtype[dtype]
    M, N = 0, 0
    if atype == "v":
        D = struct.unpack('i', fd.read(4))[0]
        str_bytes = fd.read(D * sizeof)
    else:
        dim_bytes = fd.read(8)
        M, N = struct.unpack('ii', dim_bytes)
        str_bytes = fd.read(M * N * sizeof)
    arr = np.fromstring(str_bytes, dtype=np_dtype)
    return arr if atype == "v" else arr.reshape(M, N)


def _parse_scripts(scp):
    index_addr = dict()
    line = 0
    with open(scp, "r") as f:
        for raw_line in f:
            toks = raw_line.strip().split()
            line += 1
            if len(toks) != 2:
                raise RuntimeError(
                    "Content format error in {}:{:d}, {}".format(
                        scp, line, raw_line))
            key, value = toks
            index_toks = value.split(":")
            if len(index_toks) != 2:
                raise RuntimeError("Value format error in {}:{:d}: {}".format(
                    scp, line, raw_line))
            obj, shift = index_toks
            if key in index_addr:
                raise ValueError("Duplicated key \'{}\' exists in {}".format(
                    key, index_addr))
            index_addr[key] = (obj, int(shift))
    return index_addr


class BinaryWriter(object):
    """
    Binary object writer
    """

    def __init__(self, obj, scp=None):
        self.obj_str, self.scp_str = obj, scp
        if not self.obj_str:
            raise RuntimeError("BinaryWriter: Seems got empty object")

    def __enter__(self):
        self.obj = open(self.obj_str, "wb")
        self.scp = None if not self.scp_str else open(self.scp_str, "w")
        return self

    def __exit__(self, *args):
        self.obj.close()
        if self.scp:
            self.scp.close()

    def write(self, key, ndarray):
        if not isinstance(ndarray, np.ndarray):
            raise TypeError("Expect 2nd args as numpy's ndarray, "
                            "but got type {}".format(type(ndarray)))
        if not isinstance(key, str):
            raise TypeError("Expect 1st args as python str object, "
                            "but got type {}".format(type(key)))
        # write key
        self.obj.write(str.encode(key + " "))
        if self.scp:
            offset = self.obj.tell()
        # write value
        _serialize(self.obj, ndarray)
        if self.scp:
            records = "{0}\t{1}:{2}\n".format(key, self.obj_str, offset)
            self.scp.write(records)


class BinaryObjectReader(object):
    """
    BinaryObjectReader: sequential access only
    """

    def __init__(self, obj):
        self.obj_str = obj

    def _next_key(self, fd):
        key = ""
        while True:
            c = bytes.decode(fd.read(1))
            if c == " " or c == "":
                break
            key += c
        return key

    def __iter__(self):
        with open(self.obj_str, "rb") as fd:
            while True:
                key = self._next_key(fd)
                if not key:
                    break
                obj = _deserialize(fd)
                yield key, obj


class BinaryScriptReader(object):
    """
    BinaryScriptReader: allows sequential/random access
    """

    def __init__(self, scp):
        self.index_addr = _parse_scripts(scp)
        self.mgr = dict()

    def __len__(self):
        return len(self.index_addr)

    def __contains__(self, key):
        return key in self.index_addr

    def __getitem__(self, key):
        return self._load(key)

    def __iter__(self):
        for key in self.index_addr:
            yield key, self._load(key)

    def _open(self, obj):
        if obj not in self.mgr:
            self.mgr[obj] = open(obj, "rb")
        return self.mgr[obj]

    def _load(self, key):
        obj, addr = self.index_addr[key]
        fd = self._open(obj)
        return _deserialize(fd, addr)

    def keys(self):
        return self.index_addr.keys()


def foo(N):
    import random
    import time
    D = 513
    frames = [random.randint(100, 200) for _ in range(N)]
    start = time.time()
    with BinaryWriter(
            "foo.uncompress.bin", scp="foo.uncompress.scp") as writer:
        for i, t in enumerate(frames):
            arr = np.random.rand(t, D)
            writer.write("utt-{:d}".format(i + 1), arr)
    print("#cost uncompress write: {:.2f}s".format(time.time() - start))
    reader = BinaryObjectReader("foo.uncompress.bin")
    start = time.time()
    for _, _ in reader:
        pass
    print("#cost uncompress read: {:.2f}s".format(time.time() - start))


if __name__ == "__main__":
    foo(200)
