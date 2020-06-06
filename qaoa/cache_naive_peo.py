import sys
sys.path.append('..')
sys.path.append('./qaoa')

import pyrofiler as prof
from multiprocessing.dummy import Pool
from multiprocessing import Pool
import utils_qaoa as qaoa
import utils
import numpy as np
import qtree
import copy
from tqdm import tqdm
from utils_mproc import *

pool = Pool(processes=20)
experiment_name = 'peos_skylake_qbb_30-45'

sizes = np.arange(30,46)

tasks = [qaoa.get_test_expr_graph(s, 1) for s in sizes]
graphs =     [g for g, _ in tasks]
qbit_sizes = [N for _, N in tasks]

print('Qubit sizes', qbit_sizes)

peos_n = pool.map(n_peo, graphs)
peos, nghs = zip(*peos_n)
peos_n = np.array(peos_n)
np.save(f'cached_data/{experiment_name}_peos_n', peos_n)

with prof.timing('Get full costs naive'):
    costs = pool.map(get_cost, zip(graphs, peos))

np.save(f'cached_data/{experiment_name}_costs', costs)


