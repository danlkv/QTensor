#!/bin/bash -l
#
echo "[entry.sh] JOB $PBS_JOBID Start. PARAM_P=$PARAM_P RANKS=$RANKS"
module load conda cray-mpich cudatoolkit-standalone
conda activate

cd $PBS_O_WORKDIR
echo "[entry.sh] Current workdir $PWD"
echo "[entry.sh] Hostname: `hostname`"
echo "[entry.sh] Parameter p: $PARAM_P"
echo "[entry.sh] Ranks: $RANKS"
export CUDA_HOME=/soft/compilers/nvidia/Linux_x86_64/2022/cuda/11.0
export PARAM_P

time mpiexec -n $RANKS --ppn 4 ./scripts/large_run.py
echo "[entry.sh] JOB $PBS_JOBID Done."
