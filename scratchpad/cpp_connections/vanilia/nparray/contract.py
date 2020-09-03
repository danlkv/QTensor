import tcontract
import sys
import numpy as np
import time

def contract():
    try:
        N = int(sys.argv[1])
    except LookupError:
        N = 0

    n, m, k = 2+N, 3+N, 4+N
    A, B = np.random.randn(n, m), np.random.randn(n, k)

    C = np.empty((n, m, k))
    size = sys.getsizeof(C)
    print('Result size = {C_size:e} bytes'.format(C_size=size))

    start = time.time()
    tcontract.triple_loop_contract(A, B, C)
    print('cpp contract time', time.time() - start)

    start = time.time()
    C_einsum =np.einsum('ij,ik -> ijk', A, B)
    print('einsum contract time', time.time() - start)
    assert np.array_equal(C_einsum, C)

contract()
