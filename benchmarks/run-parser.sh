#!/bin/bash -e
# File: run-parser.sh
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

# parse logs produced by parse-log.py for plotting

PREFIX=$1

python2 parse-log.py "$PREFIX".log > "$PREFIX".data

cat "$PREFIX".data | awk ' ($1==1) {print $0}' > "$PREFIX"-b1.data
cat "$PREFIX".data | awk ' ($1==64) {print $0}' > "$PREFIX"-b64.data

cat "$PREFIX"-b1.data | paste - - - - | \
	awk '($4==3) {print $2" "$6" "$12" "$18" "$24}' > "$PREFIX"-b1c3.data
cat "$PREFIX"-b1.data | paste - - - - | \
	awk '($4==64) {print $2" "$6" "$12" "$18" "$24}' > "$PREFIX"-b1c64.data

cat "$PREFIX"-b64.data | paste - - - - | \
	awk '($4==3) {print $2" "$6" "$12" "$18" "$24}' > "$PREFIX"-b64c3.data
cat "$PREFIX"-b64.data | paste - - - - | \
	awk '($4==64) {print $2" "$6" "$12" "$18" "$24}' > "$PREFIX"-b64c64.data

