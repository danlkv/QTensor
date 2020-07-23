import sys
from benchmark import peo_solvers

number = int(sys.argv[1])
peo_solvers.run_peo_benchmarks([number])
