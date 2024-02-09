#!/bin/bash
#

NODES=2
RANKS=$(( NODES * 4 ))
QUEUE=debug-scaling
WALLTIME=40:00

qsub -l select=$NODES:system=polaris:ncpus=32:ngpus=4:gputype=A100,walltime=$WALLTIME,filesystems=home \
    -q $QUEUE -AQTensor \
    -v RANKS=$RANKS,PARAM_P=$PARAM_P \
    -o job_out.output -e job_out.output \
    ./scripts/polaris/entry.sh

echo -e "===========\nNew job with NODES=$NODES, PARAM_P=$PARAM_P submitted.\n" >> job_out.output
sleep 1.5
tail -f job_out.output

