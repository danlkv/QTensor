#!env python
import tcontract
import sys
import numpy as np
try:
    from opt_einsum import contract as opt_einsum
except ImportError:
    opt_einsum = None

import time
from pyrofiler import Profiler

def random_complex(*shape):
    return np.random.randn(*shape) + 1j*np.random.randn(*shape)

def stats_callback(elapsed_time, description, cop, flop):
    print(f'{description}: Elapsed time={round(elapsed_time,3)} COPS={cop/elapsed_time:e} FLOPS={flop/elapsed_time:e}')

def contract():
    try:
        N = int(sys.argv[1])
    except LookupError:
        N = 0
    prof = Profiler(callback=stats_callback)

    n, m, k = 2+N, 3+N, 4+N
    A, B = np.random.randn(n, m), np.random.randn(n, k)

    C = np.empty((n, m, k))
    cop = C.size
    flop = 6*C.size
    size = sys.getsizeof(C)
    print('Result size = {C_size:e} bytes'.format(C_size=size))

    with prof.timing('Einsum', cop=cop, flop=flop):
        C_einsum =np.einsum('ij,ik -> ijk', A, B)


    with prof.timing('MKL', cop=cop, flop=flop):
        tcontract.mkl_contract(A, B, C)

    if opt_einsum:
        with prof.timing('Opt Einsum', cop=cop, flop=flop):
            _ = opt_einsum('ij,ik -> ijk', A, B)#, backend='torch')

    assert np.array_equal(C_einsum, C)


def contract_sum():
    try:
        N = int(sys.argv[1])
    except LookupError:
        N = 100
    prof = Profiler(callback=stats_callback)

    n, m, k, f = N, 1+N, 2+N, 3+N
    #k = 2
    print('Summation size:', k)
    A, B = random_complex(k, f, m), random_complex(k, f, n)

    C = np.empty((f, m, n), dtype=np.complex128)
    flop = 6*C.size * (2*k - 1)
    cop = C.size * (k - 1)
    size = sys.getsizeof(C)
    print('Result size = {C_size:e} bytes'.format(C_size=size))

    with prof.timing('Einsum', cop=cop, flop=flop):
        C_einsum =np.einsum('kfm,kfn -> fmn', A, B)

    with prof.timing('MKL contract_summ', cop=cop, flop=flop):
        tcontract.mkl_contract_sum(A, B, C)

    assert np.allclose(C_einsum, C)

if __name__=="__main__":
    print('**With summ**')
    contract_sum()
    print()
    print('**Just multiply**')
    contract()
    print()
