#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: tensorio.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
import tensorflow as tf

def read_value(f):
    meta = f.readline().strip().split()
    if not meta:
        raise OSError("EOF!")
    name = meta[0]
    shape = map(int, meta[1:])
    nele = np.prod(shape)
    data = f.read(nele * 4)
    arr = np.frombuffer(data, dtype='float32')
    arr = arr.reshape(shape)
    print(name, shape)
    return name, arr

def read_to_tf_var(f, cnt):
    ret = []
    for k in range(cnt):
        name, arr = read_value(f)
        var = tf.Variable(arr, name=name)
        ret.append(var)
    return ret

def write_value(f, name, arr):
    assert arr.dtype == 'float32'
    shape = arr.shape
    f.write(name + " ")
    f.write(" ".join(map(str, shape)))
    f.write("\n")
    buf = arr.tobytes()
    f.write(buf)

if __name__ == '__main__':
    import sys
    f = open(sys.argv[1])
    while True:
        read_value(f)
