import numpy as np
import ctypes
from ctypes import *
import random
#from qtensor.tools.lazy_import import cupy as cp
import cupy as cp
import time
import torch

import cuszp

from pathlib import Path

def cuszp_device_compress(oriData, absErrBound,threshold):

    oriData = oriData.flatten()
    x = torch.as_tensor(oriData, device='cuda')
    
    ori_real = x.real
    ori_imag = x.imag
    x = x.contiguous()
    x = torch.cat((ori_real, ori_imag))
    x = torch.flatten(x)
    bitmap = None
    d = torch.max(x) - torch.min(x)
    d = d.item()
    absErrBound = float(absErrBound*(d))
    threshold = threshold*(d)
    truth_values = torch.abs(x)<=threshold
    x[truth_values] = 0.0

    o_bytes = cuszp.compress(x, absErrBound, "rel")
    outSize = o_bytes.numel()*o_bytes.element_size()

    return (o_bytes,bitmap, absErrBound), outSize


def cuszp_device_decompress(nbEle, cmpBytes):

    (cmpBytes, bitmap, absErrBound) = cmpBytes

    newData = cuszp.decompress(
        cmpBytes,
        nbEle,
        cmpBytes.numel()*cmpBytes.element_size(),
        absErrBound,
        "rel",
    )

    arr = cp.asarray(newData)
    res = arr
    c_res = cp.zeros(int(nbEle/2), np.complex64)
    c_res.real = res[0:int(nbEle/2)]
    c_res.imag = res[int(nbEle/2):]

    return (c_res, None)

### Example of device compress/decompress wrapper usage
class Comp():
    def __init__(self):
        self.name = "dummy"

def free_compressed(ptr):
    p_ptr = ctypes.addressof(ptr)
    p_int = ctypes.cast(p_ptr, ctypes.POINTER(ctypes.c_uint64))
    decomp_int = p_int.contents
    #cp.cuda.runtime.free(decomp_int.value)


if __name__ == "__main__":
    
    DATA_SIZE = int(1024*64)
    MAX_D = 10.0
    MIN_D = -10.0
    RANGE = MAX_D - MIN_D
    r2r_threshold = 0.01
    r2r_error = 0.01
    ranga_vr = RANGE
    in_vector = np.zeros((DATA_SIZE,))
    for i in range(0,int(DATA_SIZE/4)):
        in_vector[i] = 0.0
    for i in range(int(DATA_SIZE/4), int(2*DATA_SIZE/4)):
        in_vector[i] = 5.0
    for i in range(int(2*DATA_SIZE/4), int(3*DATA_SIZE/4)):
        in_vector[i] = random.uniform(MIN_D, MAX_D)
    for i in range(int(3*DATA_SIZE/4), int(3*DATA_SIZE/4)+6):
        in_vector[i] = -7.0
    for i in range(int(3*DATA_SIZE/4)+6, DATA_SIZE):
        in_vector[i] = 0.001

    print(DATA_SIZE)
    in_vector = in_vector.astype('complex64')
    in_vector_gpu = cp.asarray(in_vector)
    
    #in_vector_gpu = cp.asarray(in_vector)
    # variable = ctypes.c_size_t(0)
    # outSize = ctypes.pointer(variable)
    for i in range(2):
        s_time = time.time()
        o_bytes, outSize = cuszp_device_compress(in_vector_gpu, r2r_error, r2r_threshold)
        print("Time python: "+str(time.time()-s_time))
        print(outSize)
        print("Compress Success...starting decompress ")
        comp = Comp()

        s_time = time.time()
        (d_bytes,ptr )= cuszp_device_decompress(DATA_SIZE*2, o_bytes)
        #free_compressed(o_bytes[0])
        #cp.cuda.runtime.free(d_bytes)
        print("Time python: "+str(time.time()-s_time))
    #for i in d_bytes:
    #    print(i)
        print("Decompress Success")
