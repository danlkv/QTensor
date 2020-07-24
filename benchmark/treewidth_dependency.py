import sys
from peo_solvers import run_treewidth_dependency_benchmarks

number = int(sys.argv[1])
run_treewidth_dependency_benchmarks([number])
