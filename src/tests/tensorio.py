#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: tensorio.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
import tensorflow as tf

def read_value(f):
    meta = f.readline().strip().split()
    name = meta[0]
    shape = map(int, meta[1:])
    nele = np.prod(shape)
    data = f.read(nele * 4)
    arr = np.fromstring(data, dtype='float32')
    arr = arr.reshape(shape)
    print name, shape
    return name, arr

def read_to_tf_var(f, cnt):
    ret = []
    for k in range(cnt):
        name, arr = read_value(f)
        var = tf.Variable(arr, name=name)
        ret.append(var)
    return ret
