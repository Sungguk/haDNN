#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: test-vgg16.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import numpy as np
import os
import argparse
import cPickle as pkl
import cv2

from tensorpack.train import TrainConfig, start_train
from tensorpack.predict import PredictConfig, get_predict_func
from tensorpack.models import *
from tensorpack.utils import *
from tensorpack.tfutils import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.callbacks import *
from tensorpack.dataflow import *
from tensorpack.dataflow.dataset import ILSVRCMeta

class Model(ModelDesc):
    def _get_input_vars(self):
        return [InputVar(tf.float32, (None, 224, 224, 3), 'input'),
                InputVar(tf.int32, (None,), 'label') ]

    def _get_cost(self, inputs, is_training):
        is_training = bool(is_training)
        keep_prob = tf.constant(0.5 if is_training else 1.0)

        image, label = inputs

        # 224
        l = Conv2D('conv1_1', image, out_channel=64, kernel_shape=3)
        l = Conv2D('conv1_2', l, out_channel=64, kernel_shape=3)
        l = MaxPooling('pool1', l, 2, stride=2, padding='VALID')
        # 112

        l = Conv2D('conv2_1', l, out_channel=128, kernel_shape=3)
        l = Conv2D('conv2_2', l, out_channel=128, kernel_shape=3)
        l = MaxPooling('pool2', l, 2, stride=2, padding='VALID')
        # 56

        l = Conv2D('conv3_1', l, out_channel=256, kernel_shape=3)
        l = Conv2D('conv3_2', l, out_channel=256, kernel_shape=3)
        l = Conv2D('conv3_3', l, out_channel=256, kernel_shape=3)
        l = MaxPooling('pool3', l, 2, stride=2, padding='VALID')
        # 28

        l = Conv2D('conv4_1', l, out_channel=512, kernel_shape=3)
        l = Conv2D('conv4_2', l, out_channel=512, kernel_shape=3)
        l = Conv2D('conv4_3', l, out_channel=512, kernel_shape=3)
        l = MaxPooling('pool4', l, 2, stride=2, padding='VALID')
        # 14

        l = Conv2D('conv5_1', l, out_channel=512, kernel_shape=3)
        l = Conv2D('conv5_2', l, out_channel=512, kernel_shape=3)
        l = Conv2D('conv5_3', l, out_channel=512, kernel_shape=3)
        l = MaxPooling('pool5', l, 2, stride=2, padding='VALID')
        # 7

        l = FullyConnected('fc6', l, 4096)
        l = tf.nn.dropout(l, keep_prob)
        l = FullyConnected('fc7', l, 4096)
        l = tf.nn.dropout(l, keep_prob)
        logits = FullyConnected('fc8', l, out_dim=1000, summary_activation=False, nl=tf.identity)
        prob = tf.nn.softmax(logits, name='output')

        y = one_hot(label, 1000)
        cost = tf.nn.softmax_cross_entropy_with_logits(logits, y)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        # compute the number of failed samples, for ValidationError to use at test time
        wrong = tf.not_equal(
            tf.cast(tf.argmax(prob, 1), tf.int32), label)
        wrong = tf.cast(wrong, tf.float32)
        nr_wrong = tf.reduce_sum(wrong, name='wrong')

        return cost

def run_test(path, input):
    param_dict = np.load(path).item()

    pred_config = PredictConfig(
        model=Model(),
        input_data_mapping=[0],
        session_init=ParamRestore(param_dict),
        output_var_names=['output:0', 'pool5/MaxPool:0']
    )
    predict_func = get_predict_func(pred_config)

    im = cv2.imread(input)
    assert im is not None
    im = im.astype('float32')
    im = cv2.resize(im, (224, 224)).reshape((1,224,224,3))
    im = im - 110
    raw_out = predict_func([im])
    tfout = raw_out[1][0]

    from tensorio import read_value
    dumpf = 'dump.tensortxt'
    with open(dumpf) as f:
        name, arr = read_value(f)
    os.unlink(dumpf)
    hout = arr[:,:,:,0]
    diff = hout - tfout
    maxdiff = np.abs(diff).max()
    print "Diff:", maxdiff
    assert maxdiff < 1e-3
    return

    prob = raw_out[0][0]
    ret = prob.argsort()[-10:][::-1]
    print ret
    meta = ILSVRCMeta().get_synset_words_1000()
    print [meta[k] for k in ret]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0',
                        help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--input', help='an input image', required=True)
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    ret = os.system('./test-vgg16.bin vgg16.tensortxt {}'.format(args.input))
    assert ret == 0
    run_test('vgg16.npy', args.input)
