#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: benchmark.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import sys, os, time

cmd_prefix = sys.argv[1]    # a command that accepts arguments B HW K Cin Cout

def test(b, cin, cout, hw, k):
    cmd = cmd_prefix + " {b} {hw} {hw} {k} {cin} {cout}".format(**locals())
    log = "B={b},HW={hw},K={k},Cin={cin},Cout={cout}".format(**locals())
    sys.stdout.write(log + '\n')
    sys.stdout.flush()
    ret = os.system(cmd)
    if ret != 0:
        raise ret
    time.sleep(1)

B = [64]
Cin = [64]
Cout = [64]
HW = [8, 15, 16, 17, 28, 30, 32, 48, 58, 60, 64, 80, 124, 128, 150, 200]
K = [3, 5, 7, 9]

for b in B:
    for cin in Cin:
        for cout in Cout:
            for hw in HW:
                for k in K:
                    test(b, cin, cout, hw, k)

