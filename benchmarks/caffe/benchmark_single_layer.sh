#!/bin/bash -e
# File: run.sh
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

NETWORK=./single_layer.prototxt
# CAFFE_ROOT=$HOME/software/caffe-cpu/
# CAFFE_ROOT=$HOME/Developer/15618/caffe
if [ "$CAFFE_ROOT" != "" ]; then
  CAFFE_ROOT=$CAFFE_ROOT
fi
if [ "$CAFFE_ROOT" = "" ]; then
  echo "Error: CAFFE_ROOT is not set."
  exit 1
fi

CAFFE_BIN=$CAFFE_ROOT/build/tools/caffe
export LD_LIBRARY_PATH=/usr/lib/openblas-base

mkdir -p results;
rm -f results/*;

for B in 1 64 128; do
    for H in 14 30 62 126 224; do
        W=$H
        for k in 3 5 7; do
        	export B=$B
        	export H=$H
        	export W=$W
        	export k=$k
        	export cin=3
        	export cout=128
			# http://stackoverflow.com/questions/2914220/bash-templating-how-to-build-configuration-files-from-templates-with-bash
        	echo "Running $B.$H.$k.$cin"
			perl -p -e 's/\$\{([^}]+)\}/defined $ENV{$1} ? $ENV{$1} : $&/eg' < single_layer.prototxt.template > single_layer.prototxt
			$CAFFE_BIN time -model=$NETWORK -iterations=5 2>&1 | grep "Average Forward pass" | awk '{print $8}' | tee results/output_`basename $NETWORK`.$B.$H.$W.$k.$cin.$cout.log
        	export cin=128
        	export cout=128
        	echo "Running $B.$H.$k.$cin"
			perl -p -e 's/\$\{([^}]+)\}/defined $ENV{$1} ? $ENV{$1} : $&/eg' < single_layer.prototxt.template > single_layer.prototxt
			$CAFFE_BIN time -model=$NETWORK -iterations=5 2>&1 | grep "Average Forward pass" | awk '{print $8}' | tee results/output_`basename $NETWORK`.$B.$H.$W.$k.$cin.$cout.log
        done
    done
done
done
