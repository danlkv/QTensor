#!/bin/bash
qsub -t 300 -n 5 -q skylake_8180 --cwd ~ $PWD/entry.sh
