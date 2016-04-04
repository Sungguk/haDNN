#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: test-fft.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from tensorio import read_value
import scipy.signal as signal
import numpy as np

fin = open('../param.tensortxt')
_, img = read_value(fin)
_, weight = read_value(fin)
_, halide_conv_out = read_value(fin)
_, halide_fft_out = read_value(fin)

scipy_conv_out_inv = signal.convolve2d(img, weight, mode='same')
weight = weight[::-1, ::-1]
scipy_conv_out = signal.convolve2d(img, weight, mode='same')


from IPython import embed; embed()
