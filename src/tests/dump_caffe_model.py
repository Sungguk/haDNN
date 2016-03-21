#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dump_caffe_model.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from argparse import ArgumentParser
import operator

from tensorpack.utils.loadcaffe import load_caffe
from tensorio import write_value

def get_args():
    parser = ArgumentParser()
    parser.add_argument('model', help='input caffe model prototxt')
    parser.add_argument('weights', help='input caffe model weights')
    parser.add_argument('output', help='output tensortxt')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    params = load_caffe(args.model, args.weights)

    fout = open(args.output, 'wb')
    for name, val in sorted(params.items(), key=operator.itemgetter(0)):
        if 'conv' in name and '/W' in name:
            # hwio to iohw
            val = val.transpose(2, 3, 0, 1)
        # TODO transpose fc?
        write_value(fout, name, val)
    fout.close()

if __name__ == '__main__':
    main()
