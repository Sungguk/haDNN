#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: test-conv.py
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
        print(time.time())
        result = result.eval()
        print(time.time())
        diff = np.max(result - output)
        assert diff < 1e-5, diff

def test_hwcn():
    f = open('conv_hwcn_out.txt', 'r')
    W1, b1, W2, b2, input = read_to_tf_var(f, 5)
    W1 = tf.transpose(W1, (2,3,0,1))
    W2 = tf.transpose(W2, (2,3,0,1))
    _, output = read_value(f)
    # W: in, out, h, w

    input = tf.transpose(input, (3, 0, 1, 2))

    #W1 = tf.ones(W1.get_shape())
    #W2 = tf.ones(W2.get_shape())

    l = tf.nn.conv2d(input, W1, strides=[1,1,1,1],
                           padding='SAME')
    l = tf.nn.bias_add(l, b1)

    l = tf.nn.conv2d(l, W2, strides=[1,1,1,1],
                           padding='SAME')
    l = tf.nn.bias_add(l, b2)
    # nhwc to hwcn
    result = tf.transpose(l, (1, 2, 3, 0))
    sess = tf.Session()
    with sess.as_default():
        tf.initialize_all_variables().run()
        print(time.time())
        result = result.eval()
        print result.shape, output.shape
        print(time.time())
        diff = np.max(np.abs(result - output))
        print("Diff: ", diff)
        if diff > 1e-5:
            print output / result
            from IPython import embed; embed()


if __name__ == '__main__':
    test_hwcn()
