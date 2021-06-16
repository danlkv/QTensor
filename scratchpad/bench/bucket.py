
import numpy as np
import random
from itertools import accumulate
import pyrofiler
import opt_einsum
import sys
import os, psutil
import fire

import qtree
import qtensor
try:
    import ctf
except ImportError:
    print('Can\'t import ctf')

ELEMENT_SIZE = np.random.randn(1).itemsize

CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def get_backend(backend):
    return {
        'mkl': qtensor.contraction_backends.CMKLExtendedBackend
        , 'einsum': qtensor.contraction_backends.NumpyBackend
        , 'tr_einsum': qtensor.contraction_backends.TransposedBackend
        , 'torch': qtensor.contraction_backends.TorchBackend
    }[backend]

@pyrofiler.mem_util('memory to contract')
def contract_random(contraction, dim=2, backend='opt_einsum'):
    inp, out = contraction.split('->')
    ixs = inp.split(',')
    all_chars = qtensor.utils.merge_sets(set(x) for x in ixs)
    summed_chars = all_chars - set(out)
    print([len(x) for x in ixs], len(out))
    tensors = [np.random.randn(*[dim]*len(s)) for s in ixs]
    bucket = [qtree.optimizer.Tensor('name', list(ix), data=data) for ix, data in zip(ixs, tensors)]
    print(f"Contracting {len(tensors)} tensors")
    print('Bucket', bucket)
    print('Contracted indices', summed_chars)
    backend = get_backend(backend)()
    with pyrofiler.timing('time to contract:'):
        r = backend.process_bucket_merged(summed_chars, bucket)
    print('Done')
    print(r.data.ravel()[0])
    return r


def test_mem(K, C=4, d=2, s=10, backend='einsum', seed=10, expr=None):
    """
    Performs a contraction that
    does not include hard optimization task.
    Majority of indices are in ``d`` tensors, and other ``s`` tensors are
    small

    Args:
        K: total number of indices in result
        C: contracted indices
        d: number of dominating tensors
        s: number of small tensors
    """
    if expr is not None:
        contract_random(expr, backend=backend)
        return

    np.random.seed(seed)
    random.seed(seed)
    if K+C>len(CHARS):
        raise Exception('Too many indices')
    if C + np.ceil(K/d) > K:
        raise Exception('Too few indices')

    #-- Memory estimations
    process = psutil.Process(os.getpid())
    overhead = process.memory_info().rss
    print('Overhead:', overhead)
    print('Should take:', ELEMENT_SIZE*2**K, 'Bytes')
    print('Total expected Memory:', overhead+ELEMENT_SIZE*2**K, 'Bytes')
    #--

    all_indices = list(CHARS[:K+C])
    contracted_indices = list(np.random.permutation(list(all_indices)[:C]))
    result_indices = list(set(all_indices) - set(contracted_indices))


    exp_dom_size = len(result_indices)/d
    # The sizes should be result_indices/d on average
    while True:
        dom_sizes = [ min(K-C, np.random.poisson(lam=exp_dom_size))
                       for _ in range(d)
                      ]
        if sum(dom_sizes) > K-1:
            break

    # Need to satisfy two constraints:
    #   1. each result index should be in at least one tensor
    #   2. size of a single tensor should be smaller than total result size

    # Create an array that is of size sum(sizes) which contains all indices.
    # Then take consecutive chunks of needed sizes from it
    draw_from_ = list(np.random.permutation((result_indices*d)[:sum(dom_sizes)]))
    starts_ = list(accumulate(dom_sizes))
    dom_ix = [
        contracted_indices + draw_from_[st-s:st] for st, s in zip(starts_, dom_sizes)
    ]
    print('large sizes', [len(x) for x in dom_ix])

    exp_small_sizes = 2
    sm_sizes = [ min(K, np.random.poisson(lam=exp_small_sizes))
                   for _ in range(s)
                  ]
    sm_ix = [
        [np.random.choice(contracted_indices)] +
        list(np.random.permutation(result_indices)[:s])
        for s in sm_sizes
    ]

    contraction = ','.join(
        ''.join(ix) for ix in sm_ix + dom_ix
    ) + '->' + ''.join(result_indices)

    print('contraction:', contraction)

    contract_random(contraction, backend=backend)

if __name__=="__main__":
    fire.Fire(test_mem)

