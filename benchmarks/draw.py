#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: draw.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import sys
import numpy as np
import pandas as pd

def read_file(filename, prefix):
    ret = []
    with open(filename) as f:
        for line in f:
            line = map(float, line.strip().split())
            ret.append(line)
    arr = np.array(ret)
    return pd.DataFrame(arr[:,1:],
            index=arr[:,0].astype('int32'),
            columns=['{} k={}'.format(prefix, k) for k in [3,5,7,9]])

def read():
    #f_caffe = read_file('caffe-th1b64c64.data', 'Caffe')
    #f_normal = read_file('normal-th1b64c64.data', 'Conv')
    #f_fft = read_file('fft-th1b64c64.data', 'FFT')
    f_caffe = read_file('caffe-b64c64.data', 'Caffe')
    f_normal = read_file('normal-b64c64.data', 'Conv')
    f_fft = read_file('fft-b64c64.data', 'FFT')
    datas = [f_caffe, f_normal, f_fft]
    joined = pd.concat(datas, axis=1)
    return joined

def print_data(data, args):
    args = args.strip().split(' ')
    filtered = []
    for arg in args:
        if arg.startswith('c'):
            filtered.append(data['Caffe k={}'.format(arg[1])])
        elif arg.startswith('n'):
            filtered.append(data['Conv k={}'.format(arg[1])])
        elif arg.startswith('f'):
            filtered.append(data['FFT k={}'.format(arg[1])])
        else:
            raise arg
    filtered = np.asarray(filtered)
    xs = np.asarray(data.index)
    for idx, x in enumerate(xs):
        print x,
        print ' '.join(map(str, filtered[:, idx]))



if __name__ == '__main__':
    data = read()
    print_data(data, sys.argv[1])
