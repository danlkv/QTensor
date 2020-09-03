import tcontract
import sys
import numpy as np

def contract():
    try:
        N = int(sys.argv[1])
    except LookupError:
        N = 24

    n, m, k = 2, 3, 3
    A, B = np.random.randn(n, m), np.random.randn(n, k)
    C = np.empty((n, m, k))
    size = sys.getsizeof(C)
    print('Result size = {C_size:e} bytes'.format(C_size=size))
    tcontract.triple_loop_contract(A, B, C)

    print(C)
    C_einsum =np.einsum('ij,ik -> ijk', A, B)
    print(C_einsum)
    assert np.array_equal(C_einsum, C)

contract()
