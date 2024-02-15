#!/bin/bash
mpiexec -n 4 ./main.py simulate ./data/preprocess/mpi_debug/qaoa/3reg_N42_p4.jsonterms_Otamaki_8_M29  ./data/simulations/mpi_debug/{in_file}_cM{M}_rE{r2r_threshold}.sim --sim qtensor -M 27 --backend=cupy --compress=szx --r2r_error=5e-5 --r2r_threshold=5e-5 --mpi

