#!/bin/bash
#
./main.py generate data/circuits/qaoa/maxcut_regular_N{N}_p{p} --type=qaoa_maxcut --N=8,12,16,24,32,48,64 --p=1,2,3,4,5 --d=3
