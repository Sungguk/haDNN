#!/bin/bash -e

mkdir results;
rm results/*;

function run {
    echo "Running $1 $2 $3 $4 $5 $6 for 5 times";
    for (( i = 0; i < 5; i++ )); do
        ./benchmark.bin $1 $2 $3 $4 $5 $6 $7 | grep Realize | awk '{print $3}' >> results/$1.$2.$3.$4.$5.$6
    done
}

for type in normal fft; do
for B in 1 64 128; do
    for H in 14 30 62 126 224; do
        W=$H
        for k in 3 5 7; do
            run $type $B $H $W $k 3 128
            run $type $B $H $W $k 128 128
        done
    done
done
done