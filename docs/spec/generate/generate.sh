#!/bin/bash
command='python -m qensor.cli qaoa-energy-tw --nodes 100 --p $0 --degree $1 --graph-type erdos_renyi'
echo '{"p":[1,2], "d":[3,4]}' | ./matrix_output.py "$command"
