#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: fft-conv-poc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
import numpy as np
import scipy.signal
import numpy.fft
import cv2

image2d = np.random.rand(200, 200)
image2dp = np.pad(image2d, ((1,1),(1,1)), mode='constant')  # pad then fft
kernel = np.random.rand(3,3)

img_f = np.fft.fft2(image2dp)
krn_f = np.fft.fft2(kernel, s=image2dp.shape)
conv = np.fft.ifft2(img_f*krn_f).real

conv = conv[2:,2:]  # 2 == pad*2 = 3//2 * 2

conv2 = scipy.signal.convolve2d(image2d, kernel, mode='same', boundary='fill')
print conv
print conv2
diff = conv2 - conv
print np.abs(diff).max()
#from IPython import embed; embed()
