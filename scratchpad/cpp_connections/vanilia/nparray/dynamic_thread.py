#!env python

import os, sys
import pyrofiler
import numpy as np

ths = [1, 2, 4, 8, 16, 32, 64]
N = int(sys.argv[1])

def random_complex(*shape):
    return np.random.randn(*shape) + 1j*np.random.randn(*shape)

import os, ctypes 
def set_mkl_threads(th):
   
    try:
        import mkl
        mkl.set_num_threads(th)
        return 0
    except:
        pass 
    
    for name in [ "libmkl_rt.so", "libmkl_rt.dylib", "mkl_Rt.dll"]: 
        try: 
            mkl_rt = ctypes.CDLL(name)
            mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(th)))
            return 0 
        except:
            pass   

for thr in ths:
    print('Setting omp num threads', thr)
    os.environ['OMP_NUM_THREADS'] = str(thr)
    set_mkl_threads(thr)
    import tcontract

    n, m, k = 2+N, 3+N, 4+N
    A, B = np.random.randn(n, m), np.random.randn(n, k)
    C = np.empty((n, m, k))
    @pyrofiler.proc_count(callback=print)
    @pyrofiler.cpu_util(callback=print)
    def c():
        tcontract.mkl_contract(A, B, C)

    with pyrofiler.timing(callback=print):
        c()



