#!/bin/bash
# UG Section 2.5, page UG-24 Job Submission Options
# Add another # at the beginning of the line to comment out a line
# NOTE: adding a switch to the command line will override values in this file.
# These options are MANDATORY at ALCF; Your qsub will fail if you don't provide them.
#PBS -A QTensor
#PBS -l walltime=01:00:00
# Highly recommended 
# The first 15 characters of the job name are displayed in the qstat output:
#PBS -N QTensor
#file systems used by the job
# PBS -l filesystems=home:grand
#PBS -l filesystems=grand
#PBS -l select=2:system=polaris
# If you need a queue other than the default, which is prod (uncomment to use)
#PBS -q debug
# Controlling the output of your application
# UG Sec 3.3 page UG-42 Managing Output and Error Files
# By default, PBS spools your output on the compute node and then uses scp to move it the
# destination directory after the job finishes.  Since we have globally mounted file systems
# it is highly recommended that you use the -k option to write directly to the destination
# the doe stands for direct, output, error
#PBS -k doe
# PBS -o ./stdout.out
# PBS -e ./stderr.out
# If you want to merge stdout and stderr, use the -j option
# oe=merge stdout/stderr to stdout, eo=merge stderr/stdout to stderr, n=don't merge
#PBS -j n
# Controlling email notifications
# UG Sec 2.5.1, page UG-25 Specifying Email Notification
# When to send email b=job begin, e=job end, a=job abort, j=subjobs (job arrays), n=no mail
#PBS -m e
# Be default, mail goes to the submitter, use this option to add others (uncomment to use)

echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`

#module load conda
#conda info --envs
#conda activate /home/wberquis/miniconda3/envs/qtensor_gpu
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qtensor_gpu

module load mpiwrappers/cray-mpich-llvm
module load PrgEnv-gnu
#module load PrgEnv-nvhpc/8.3.3
#module load PrgEnv-cray
#module load cray-mpich
#module load cudatoolkit-standalone/11.8.0
#module load craype-accel-nvidia70

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4
NDEPTH=8
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

# wait

#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth ./set_affinity_gpu_polaris.sh python  /home/wberquis/repos/QTensor/scratchpad/gpu_node_parallelism.py
#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth python  /home/wberquis/repos/QTensor/qtensor/noise_simulator/comparison_mpi.py

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth ./set_affinity_gpu_polaris.sh python /home/wberquis/repos/QTensor/qtensor/qnas/qiskit_qnas/run_qtensor_qaoa_evals_gpu.py
