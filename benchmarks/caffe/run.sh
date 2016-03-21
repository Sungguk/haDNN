#!/bin/bash -e
# File: run.sh
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

NETWORK=./vgg_a.prototxt
CAFFE_ROOT=$HOME/software/caffe-cpu/
#CAFFE_ROOT=$HOME/Work/DL/caffe
CAFFE_BIN=$CAFFE_ROOT/build/tools/caffe

$CAFFE_BIN time -model=$NETWORK -iterations=5 2>&1 | tee output_`basename $NETWORK`.log

