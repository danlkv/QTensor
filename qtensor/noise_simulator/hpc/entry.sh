#!/bin/bash
source $HOME/.profile
mpiexec -n 250 python $HOME/anl/Qensor/qtensor/noise_simulator/comparison_mpi.py

