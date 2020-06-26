import os
import psutil
import numpy as np
from collections import defaultdict
from multiprocessing.dummy import Pool as ThreadPool

import pyrofiler as prof
from qtree.logger_setup_mpi import get_rank_logger 

log = get_rank_logger()

pool = ThreadPool(processes=2**6)

log = get_rank_logger()
prof = prof.Profiler()
prof_data = defaultdict(lambda:[])
def callback(res, **kw):
	descr = kw.get('description', 'prof')
	prof_data[descr].append(res)

prof.data = prof_data
prof.cb = callback

def _none_slice():
    return slice(None)

def _get_idx(x, idxs, slice_idx, shape=None):
    if shape is None:
        shape = [2]*len(idxs)
    point = np.unravel_index(slice_idx, shape)
    get_point = {i:p for i,p in zip(idxs, point)}
    if x in idxs:
        p = get_point[x]
        return slice(p,p+1)
    else:
        return _none_slice()

def _slices_for_idxs(idxs, *args, shapes=None, slice_idx=0):
    """Return array of slices along idxs"""
    slices = []
    for n, indexes in enumerate(args):
        if shapes is not None:
            shape = shapes[n]
        else:
            shape = None
        _slice = [_get_idx(x, idxs, slice_idx, shape) for x in indexes ]
        slices.append(tuple(_slice))
    return slices

def sum_first(A):
    """Summates the first index of a tensor"""
    a, idxa = A
    #print('contract result idx',result_idx)
    C = np.einsum('i...->...', a)
    return C

def contract(A, B):
    a, idxa = A
    b, idxb = B
    contract_idx = set(idxa) & set(idxb)
    result_idx = tuple(sorted(set(idxa + idxb)))
    #print('contract result idx',result_idx)
    C = np.einsum(a,idxa, b,idxb, result_idx)
    return C

def sliced_contract(x, y, idxs, num):
    slices = _slices_for_idxs(idxs, x[1], y[1], slice_idx=num)
    a = x[0][slices[0]]
    b = y[0][slices[1]]
    C = contract((a, x[1]), (b, y[1]))
    return C

def sliced_sum_first(x, idxs, num):
    slices = _slices_for_idxs(idxs, x[1], slice_idx=num)
    a = x[0][slices[0]]
    C = sum_first((a, x[1]))
    return C

def target_slice(result_idx, idxs, num):
    slices = _slices_for_idxs(idxs, result_idx, slice_idx=num)
    return slices

def get_par_rank(N):
    if N>22:
        rank = int((N-20)/2)
    else:
        rank =0
    return rank

def parallel_sum_first(A, idxa):
    x  = (A, idxa)
    result_idx = idxa[1:]
    target_shape = A.shape[1:]

    N = sum(i>1 for i in target_shape)
    rank = get_par_rank(N)
    if rank == 0:
        return sum_first(x)
    log.debug(f'Sum rank {rank} for size {N}, target shape {target_shape} with len {len(target_shape)}')
    log.debug(f'Memory stat: {psutil.virtual_memory()}, will require: {2**(N+4)}')

    par_vars = idxa[1:rank+1]
    os.global_C = np.empty(target_shape, dtype=A.dtype)

    @log.catch()
    def work(i):
        #print(f'par vars: {par_vars}, indexes: {idxa}, {idxb}')
        patch = sliced_sum_first((A,idxa), par_vars, i)
        sl = target_slice(result_idx, par_vars, i)
        os.global_C[sl[0]] = patch

    threads = 2**len(par_vars)
    _ = pool.map(work, range(threads))
    #assert np.array_equal(C , os.global_C)
    return os.global_C


@prof.timed('parallel contract')
@prof.mem('contr process memory:')
def parallel_contract(A, idxa, B, idxb):
    result_idx = tuple(sorted(set(idxa + idxb)))
    #print(f'{idxa}, {A.shape} and {idxb}, {B.shape}')
    idx_to_least_idx = {old_idx: new_idx for new_idx, old_idx
                        in enumerate(result_idx)}
    idxa = [idx_to_least_idx[x] for x in idxa]
    idxb = [idx_to_least_idx[x] for x in idxb]
    result_idx = tuple(sorted(set(idxa + idxb)))
    idx_shape_pairs = (idxa+idxb, A.shape+B.shape)
    shape_dict = {i:s for i, s in zip(*idx_shape_pairs)}

    x, y = (A, idxa), (B, idxb)
    target_shape = [shape_dict[i] for i in result_idx]
    #target_shape = [2]*len(result_idx)

    N = sum(i>1 for i in target_shape)
    rank = get_par_rank(N)
    if rank == 0:
        return contract(x, y)
    log.debug(f'Contract rank {rank} for size {N}, target shape {target_shape} with len {len(target_shape)}')
    log.debug(f'Memory stat: {psutil.virtual_memory()}, will require: {2**(N+4)}')


    par_vars = result_idx[:rank]
    os.global_C = np.empty(target_shape, dtype=A.dtype)

    @log.catch()
    def work(i):
        #print(f'par vars: {par_vars}, indexes: {idxa}, {idxb}')
        patch = sliced_contract((A,idxa), (B, idxb), par_vars, i)
        sl = target_slice(result_idx, par_vars, i)
        os.global_C[sl[0]] = patch

    threads = 2**len(par_vars)
    _ = pool.map(work, range(threads))
    #assert np.array_equal(C , os.global_C)
    return os.global_C

