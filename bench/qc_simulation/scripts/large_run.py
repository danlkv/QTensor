#!/bin/bash
./main.py simulate ./data/preprocess/sc23/qaoa/3reg_N72_p3.jsonterms_Otamaki_30_M30_M24.json ./data/simulations/sc23/large/{in_file}_cM{M}_rE{r2r_threshold}.sim --sim qtensor -M 27 --backend=cupy --compress=szx --r2r_error=5e-4 --r2r_threshold=5e-4 --mpi

