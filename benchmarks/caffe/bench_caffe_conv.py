#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: bench_caffe_conv.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
import sys, os
CAFFE_ROOT="/home/wyx/software/caffe-cpu/"
#CAFFE_ROOT="/home/wyx/System/installation/caffe/"
os.environ['LD_LIBRARY_PATH']='/opt/OpenBLAS/lib'

TEMPLATE = """
name: "haha"
input: "data"
input_dim: {B}
input_dim: {Cin}
input_dim: {H}
input_dim: {W}
force_backward: false
layers {{
  name: "conv1"
  type: CONVOLUTION
  bottom: "data"
  top: "conv1"
  convolution_param {{
    num_output: {Cout}
    kernel_size: {k}
    pad: {pad}
    stride: 1
  }}
}}
"""
B, H, W, k, Cin, Cout = sys.argv[1:]
pad = int(k) / 2
proto = TEMPLATE.format(**locals())
#print proto

with open('proto.prototxt', 'w') as f:
    f.write(proto)

CAFFE_BIN = os.path.join(CAFFE_ROOT, 'build/tools/caffe')
os.system('{} time -model=proto.prototxt -iterations=10 2>&1 | grep "Average Forward pass"'.format(CAFFE_BIN))
