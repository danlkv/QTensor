import tcontract
import sys
import torch as t
import numpy as np
from opt_einsum import contract as opt_einsum

import time
from pyrofiler import Profiler

def random_complex(*shape):
    return np.random.randn(*shape) + 1j*np.random.randn(*shape)

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

    with prof.timing('Einsum', ops=Ops):
        C_einsum =np.einsum('ij,ik -> ijk', A, B)


    with prof.timing('MKL', ops=Ops):
        tcontract.mkl_contract(A, B, C)

    with prof.timing('Opt Einsum', ops=Ops):
        _ = opt_einsum('ij,ik -> ijk', t.Tensor(A), t.Tensor(B), backend='torch')

    assert np.array_equal(C_einsum, C)


def contract_sum():
    try:
        N = int(sys.argv[1])
    except LookupError:
        N = 100
    def stats_callback(elapsed_time, description, ops):
        print(f'{description}: Elapsed time={round(elapsed_time,3)} FLOPS={ops/elapsed_time:e}')
    prof = Profiler(callback=stats_callback)

    n, m, k, f = N, 1+N, 2+N, 3+N
    A, B = random_complex(k, f, m), random_complex(k, f, n)

    C = np.empty((f, m, n), dtype=np.complex128)
    Ops = C.size
    size = sys.getsizeof(C)
    print('Result size = {C_size:e} bytes'.format(C_size=size))

    with prof.timing('Einsum', ops=Ops):
        C_einsum =np.einsum('kfm,kfn -> fmn', A, B)

    with prof.timing('MKL contract_summ', ops=Ops):
        tcontract.mkl_contract_sum(A, B, C)

    assert np.allclose(C_einsum, C)

contract_sum()
contract()
