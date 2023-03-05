#!/bin/bash
#
./main.py preprocess tar://*0.txt::https://github.com/danlkv/GRCS/raw/master/inst/bristlecone/cz_v2/bris_5.tar.gz data/preprocess/bris/\{in_file\}_oalgo{O}.circ --O=greedy,rgreedy --sim=qtensor
