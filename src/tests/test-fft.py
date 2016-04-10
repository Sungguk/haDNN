#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: test-fft.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from tensorio import read_value
import scipy.signal as signal
import numpy as np
import numpy.fft as fft


def test_2d():
    fin = open('../param.tensortxt')
    _, img = read_value(fin)
    _, weight = read_value(fin)
    _, halide_conv_out = read_value(fin)
    _, halide_fft_out = read_value(fin)

    scipy_conv_out_inv = signal.convolve2d(img, weight, mode='same')
    weight = weight[::-1, ::-1]
    scipy_conv_out = signal.convolve2d(img, weight, mode='same')

    from IPython import embed; embed()

def test_fft():
    fin = open('../param.tensortxt')
    _, img = read_value(fin)
    _, halide_fft = read_value(fin)
    halide_fft = halide_fft.reshape(halide_fft.shape[:-1] + (-1, 2))
    halide_fft = halide_fft[...,0] + 1j * halide_fft[...,1]

    np_fft = fft.fft2(img, s=[16, 16])
    from IPython import embed; embed()

def test_all():
    fin = open('../param.tensortxt')

    _, img = read_value(fin)
    _, W = read_value(fin)
    _, conv_out = read_value(fin)
    _, fft_out = read_value(fin)
    #conv_out = conv_out.transpose([3,2,0,1])
    diff = conv_out - fft_out
    print diff.max(), diff.min()
    from IPython import embed; embed()

if __name__ == '__main__':
    test_all()
