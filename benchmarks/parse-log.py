#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# File: parse-log.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import sys
import re
lines = open(sys.argv[1]).readlines()

"""
parse logs produced by benchmark.py to a better formatting
"""

def parse_param(line):
    ret = re.findall('[0-9]+', line)
    assert len(ret) == 5
    return ret

def parse_time(line):
    match = re.findall('[0-9]+.[0-9]+', line)
    return float(match[-1])

nr_sample = len(lines) / 2

for idx in range(nr_sample):
    param_line = lines[idx * 2]
    perf_line = lines[idx*2+1]
    param = parse_param(param_line)
    time = parse_time(perf_line)
    print ' '.join(param), time
