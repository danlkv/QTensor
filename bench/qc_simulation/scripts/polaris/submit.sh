#!/bin/bash
#

NODES=256
RANKS=$(( NODES * 4 ))
QUEUE=prod
WALLTIME=420:00

qsub -l select=$NODES:system=polaris:ncpus=32:ngpus=4:gputype=A100,walltime=$WALLTIME,filesystems=home \
    -q $QUEUE -ACatalyst \
    -v RANKS=$RANKS,PARAM_P=$PARAM_P \
    -o job_out.output -e job_out.output \
    ./entry.sh

echo -e "===========\nNew job with NODES=$NODES, PARAM_P=$PARAM_P submitted.\n" >> job_out.output
sleep 1.5
tail -f job_out.output

