#!/bin/bash
command='python -m qtensor.cli qaoa-energy-sim --nodes 1000 --p $0 --degree $1 --graph-type random_regular --max-time 500 --n_processes=56'
echo '{"p":[2,3,4], "d":[3]}' | ./matrix_output.py "$command"
