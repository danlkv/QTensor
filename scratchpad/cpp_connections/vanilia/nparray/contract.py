import tcontract
import sys
import torch as t
import numpy as np
from opt_einsum import contract as opt_einsum

import time
from pyrofiler import Profiler

def contract():
    try:
        N = int(sys.argv[1])
    except LookupError:
        N = 0
    def stats_callback(elapsed_time, description, ops):
        print(f'{description}: Elapsed time={round(elapsed_time,3)} FLOPS={ops/elapsed_time:e}')
    prof = Profiler(callback=stats_callback)

    n, m, k = 2+N, 3+N, 4+N
    A, B = np.random.randn(n, m), np.random.randn(n, k)

    C = np.empty((n, m, k))
    Ops = C.size
    size = sys.getsizeof(C)
    print('Result size = {C_size:e} bytes'.format(C_size=size))

    start = time.time()
    with prof.timing('Triple loop', ops=Ops):
        tcontract.triple_loop_contract(A, B, C)

    with prof.timing('Einsum', ops=Ops):
        C_einsum =np.einsum('ij,ik -> ijk', A, B)

    with prof.timing('Opt Einsum', ops=Ops):
        _ = opt_einsum('ij,ik -> ijk', t.Tensor(A), t.Tensor(B), backend='torch')

    assert np.array_equal(C_einsum, C)

contract()
