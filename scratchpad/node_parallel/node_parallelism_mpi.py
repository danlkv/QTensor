import pyrofiler as pyrof
from pyrofiler.pyrofiler import Profiler
from pyrofiler import callbacks
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from multiprocessing import Pool, Array
from multiprocessing.dummy import Pool as ThreadPool
import os
from qtensor import tools
from itertools import repeat
import subprocess

sns.set_style('whitegrid')
np.random.seed(42)
# pool = ThreadPool(processes=2**7)



def _none_slice():
    return slice(None)

def _get_idx(x, idxs, slice_idx, shapes=None):
    if shapes is None:
        shapes = [2]*len(idxs)
    point = np.unravel_index(slice_idx, shapes)
    get_point = {i:p for i,p in zip(idxs, point)}
    if x in idxs:
        p = get_point[x]
        return slice(p,p+1)
    else:
        return _none_slice()

def _slices_for_idxs(idxs, *args, shapes=None, slice_idx=0):
    """Return array of slices along idxs"""
    slices = []
    for indexes in args:
        _slice = [_get_idx(x, idxs, slice_idx, shapes) for x in indexes ]
        slices.append(tuple(_slice))
    return slices
        
def get_example_task(A=8, B=10, C=7, dim1=0):
    shape1 = [2]*(A+B)
    shape2 = [2]*(A+C)
    for i in range(dim1):
        shape1[-i] = 1
        shape2[-i] = 1

    T1 = np.random.randn(*shape1)
    T2 = np.random.randn(*shape2)
    common = list(range(A))
    idxs1 = common + list(range(A, A+B))
    idxs2 = common + list(range(A+B, A+B+C))
    return (T1, idxs1), (T2, idxs2)

def contract(A, B, output = None):
    a, idxa = A
    b, idxb = B
    contract_idx = set(idxa) & set(idxb)
    result_idx = set(idxa + idxb)
    if output is not None:
        f'{output}\n contract result idx: {result_idx}'
    else: 
        f'contract result idx: {result_idx}'
    C = np.einsum(a,idxa, b,idxb, result_idx)
    return C

def sliced_contract(x, y, idxs, num, output = None):
    slices = _slices_for_idxs(idxs, x[1], y[1], slice_idx=num)
    a = x[0][slices[0]]
    b = y[0][slices[1]]
    if output is not None:
        with pyrof.timing(f'{output}\n \tcontract sliced {num}'):
            C = contract((a, x[1]), (b, y[1]), output)
    else: 
        with pyrof.timing(f'\tcontract sliced {num}'):
            C = contract((a, x[1]), (b, y[1]))
    return C

def target_slice(result_idx, idxs, num):
    slices = _slices_for_idxs(idxs, result_idx, slice_idx=num)
    return slices

def __contract_bound(A, B):
    a, idxa = A
    b, idxb = B
    contract_idx = set(idxa) & set(idxb)
    def glue_first(shape):
        sh = [shape[0] * shape[1]] + list(shape[2:])
        return sh
    
    result_idx = set(idxa + idxb)
    _map_a = {k:v for k,v in zip(idxa, a.shape)}
    _map_b = {k:v for k,v in zip(idxb, b.shape)}
    _map = {**_map_a, **_map_b}
    print(_map)
    result_idx = sorted(tuple(_map.keys()))
    target_shape = tuple([_map[i] for i in result_idx])
    
    
            
    _dimlen = len(result_idx)
    _maxdims = 22
    print('dimlen',_dimlen)
    new_a, new_b = a.shape, b.shape
    if _dimlen>_maxdims:
        _contr_dim = _dimlen - _maxdims
        print(len(new_a), len(new_b))
        for i in range(_contr_dim):
            idxa = idxa[1:]
            idxb = idxb[1:]
                    
            new_a = glue_first(new_a)
            new_b = glue_first(new_b)
            
    _map_a = {k:v for k,v in zip(idxa, a.shape)}
    _map_b = {k:v for k,v in zip(idxb, b.shape)}
    _map = {**_map_a, **_map_b}
    print(_map)
    result_idx = sorted(tuple(_map.keys()))
    print(len(new_a), len(new_b))
    a = a.reshape(new_a)
    b = b.reshape(new_b)
    print(a.shape, b.shape)
    print(idxa, idxb)
    print('btsh',result_idx, target_shape)
        
        
    C = np.einsum(a,idxa, b,idxb, result_idx)
    
    return C.reshape(*target_shape)

def __add_dims(x, dims, ofs):
    arr, idxs = x
    arr = arr.reshape(list(arr.shape) + [1]*dims)
    md = max(idxs)
    return arr, idxs + list(range(md+ofs, ofs+md+dims)) 

# def _mpi_unit(args):
#     i,x,y, par_vars, result_idx, target_shape =  args
#     patch = sliced_contract(x, y, par_vars, i)
#     sl = target_slice(result_idx, par_vars, i)
#     os.global_C[sl[0]] = patch

def _mpi_unit(args):
    output = subprocess.getoutput('echo echo')
    #output = subprocess.getoutput('echo “RANK= ${PMI_RANK} LOCAL_RANK= ${PMI_LOCAL_RANK} gpu= $((${PMI_LOCAL_RANK} % $(nvidia-smi -L | wc -l)))”')
    i,x,y, par_vars, result_idx, target_shape =  args
    patch = sliced_contract(x, y, par_vars, i, output)
    sl = target_slice(result_idx, par_vars, i)
    C_par = np.empty(target_shape)
    C_par[sl[0]] = patch
    return C_par

def parallel_contract(x, y, num_jobs, pbar = False, print_stats = True):
    # x, y = get_example_task(A=10)

    prof_seq = Profiler()
    prof_seq.use_append()

    contract_idx = set(x[1]) & set(y[1])
    result_idx = set(x[1] + y[1])

    arggen = i , 
    for i in range(1):
        _ = contract(x,y)
    for rank in range(1,7):
        with prof_seq.timing('Single thread'):
            C = contract(x,y)
        
        par_vars = list(range(rank))
        print(par_vars)
        target_shape = C.shape

        with prof_seq.timing('One patch: total'):
            i = 0
            with prof_seq.timing('One patch: compute'):
                patch = sliced_contract(x, y, par_vars, i)
            C_par = np.empty(target_shape)
            with prof_seq.timing('One patch: assign'):
                _slice = target_slice(result_idx, par_vars, i)
                C_par[_slice[0]] = patch

def parallel_contract_mpi_old(x, y, num_jobs, pbar = False, print_stats = True):
    # x, y = get_example_task(A=10)

    prof_seq = Profiler()
    prof_seq.use_append()

    contract_idx = set(x[1]) & set(y[1])
    result_idx = set(x[1] + y[1])

    i = 0
    for rank in range(1,7):
        with prof_seq.timing('Single thread'):
            C = contract(x,y)
        
        par_vars = list(range(rank))
        target_shape = C.shape
        os.global_C = np.empty(target_shape, dtype=x[0].dtype)
        # arggen = [i,x,y, par_vars, result_idx]
        # arggen = (i,x,y,par_vars,result_idx) for i in range(num_jobs)
        arggen = list(zip(repeat(i, num_jobs), repeat(x, num_jobs), repeat(y, num_jobs), repeat(par_vars, num_jobs), repeat(result_idx, num_jobs), repeat(target_shape, num_jobs)))
        result = tools.mpi.mpi_map(_mpi_unit, arggen, pbar=pbar, total=num_jobs)
        if result: 
            pass
            # result = tools.mpi.mpi_map(_mpi_unit, arggen, pbar=pbar, total=num_jobs)
            # if print_stats:
            #     tools.mpi.print_stats()
            # print(os.global_C)
            # return os.global_C
            # print(result)
            # return result

def parallel_contract_mpi(x, y, num_jobs, pbar = False, print_stats = True):
    """Get args for slicing"""
    result_idx = set(x[1] + y[1])
    #result_idx = set([i for i in x[1] if i not in set(y[1])])
    print(x[1], "\n", y[1], '\n', result_idx)
    idx_shape_pairs = (x[1] + y[1], x[0].shape + y[0].shape)
    shape_dict = {i:s for i, s in zip(*idx_shape_pairs)}
    target_shape = [shape_dict[i] for i in result_idx]
    par_vars = np.arange(1,7)
    i = 0

    """Create a set of the args to send out to workers"""
    arggen = list(zip(repeat(i, num_jobs), repeat(x, num_jobs), repeat(y, num_jobs), repeat(par_vars, num_jobs), repeat(result_idx, num_jobs), repeat(target_shape, num_jobs)))
    
    """Send out slicing jobs to workers"""
    result = tools.mpi.mpi_map(_mpi_unit, arggen, pbar=pbar, total=num_jobs)

    if result: 
        pass


if __name__ == '__main__':
    x, y = get_example_task(A=8, B = 9)
    num_nodes = 2
    num_jobs_per_node = 4
    num_jobs = num_nodes * num_jobs_per_node
    parallel_contract_mpi(x, y, num_jobs)
