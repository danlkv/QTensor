from mpi4py import MPI
import platform

w = MPI.COMM_WORLD
comm = MPI.Comm
mprank = comm.Get_rank(w)

print(f'Salutations! I am {platform.node()} rank {mprank} of {comm.Get_size(w)}')

import sys
sys.path.append('..')
sys.path.append('.')
sys.path.append('./qaoa')

import pyrofiler as prof
import utils_qaoa as qaoa
import utils
import numpy as np
from functools import partial

import qtree

s = 43
par_vars = int(sys.argv[1])
residue_len = int(sys.argv[2])

print('Using size', s)
circuit, n_qubits = qaoa.get_test_qaoa(s, 1, type='randomreg', degree=3, seed=42)
print(f'qubits count: {n_qubits}, for size {s}')
with prof.timing(f'Simulation {s} with {n_qubits} qubits'):
	result = qtree.simulator_chop.simulate_chopped(
			circuit, n_qubits
			, target_state=0
			, par_vars=par_vars
			, chop_last=residue_len
			)
result = np.array(result).reshape(-1)
if result:
	print('simulator', result.round(6))
else:
	print('Simulator returned ', result)
