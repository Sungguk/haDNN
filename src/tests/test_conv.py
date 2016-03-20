#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: test_conv.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
import numpy as np
import time
import tensorflow as tf

from tensorio import *

def test_nchw():
    f = open('conv_nchw_out.txt', 'r')
    W, b, input = read_to_tf_var(f, 3)
    _, output = read_value(f)
    # W: in, out, h, w

    conv_out = tf.nn.conv2d(input, tf.transpose(W, (2, 3, 0, 1)), strides=[1,1,1,1],
                           padding='SAME', data_format='NCHW')
    result = tf.nn.bias_add(conv_out, b, data_format='NCHW')
    sess = tf.Session()
    with sess.as_default():
        tf.initialize_all_variables().run()
        print time.time()
        result = result.eval()
        print time.time()
        diff = np.max(result - output)
        assert diff < 1e-5, diff

def test_nchw():
    f = open('conv_hwcn_out.txt', 'r')
    W, b, input = read_to_tf_var(f, 3)
    _, output = read_value(f)
    # W: in, out, h, w

    input = tf.transpose(input, (3, 2, 0, 1))
    conv_out = tf.nn.conv2d(input, tf.transpose(W, (2, 3, 0, 1)), strides=[1,1,1,1],
                           padding='SAME', data_format='NCHW')
    result = tf.nn.bias_add(conv_out, b, data_format='NCHW')
    result = tf.transpose(result, (2, 3, 1, 0))
    sess = tf.Session()
    with sess.as_default():
        tf.initialize_all_variables().run()
        print time.time()
        result = result.eval()
        print time.time()
        diff = np.max(result - output)
        print "Diff: ", diff
        assert diff < 1e-5, diff

if __name__ == '__main__':
    test_nchw()
