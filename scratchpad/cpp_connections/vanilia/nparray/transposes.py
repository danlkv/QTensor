import tcontract
import sys
import numpy as np

def large_transpose():
    try:
        N = int(sys.argv[1])
    except LookupError:
        N = 24
    arr = np.random.randn(*[2]*N)
    size = sys.getsizeof(arr)
    print('Array size = {C_size:e} bytes'.format(C_size=size))
    tcontract.print_4(arr)
    print('\n== transposed: reverse (worst case) ==')
    arr = arr.transpose(*reversed(range(N)))
    tcontract.print_4(arr)

    arr = np.random.randn(*[2]*N)
    print('\n== transposed: start (good cache efficiency) ==')
    arr = arr.swapaxes(1,0)
    tcontract.print_4(arr)

    arr = np.random.randn(*[2]*N)
    print('\n== transposed: end (low cache efficiency) ==')
    arr = arr.swapaxes(N-1,N-2)
    tcontract.print_4(arr)

large_transpose()
