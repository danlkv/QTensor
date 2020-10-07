#!/bin/bash
command='python -m qtensor.cli qaoa-energy-tw --nodes 1000 --p $0 --degree $1 --graph-type random_regular --max-time 500'
echo '{"p":[2,3,4,5], "d":[3,4]}' | ./matrix_output.py "$command"
