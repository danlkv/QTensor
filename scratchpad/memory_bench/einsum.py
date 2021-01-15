import numpy as np
from itertools import accumulate
import sys
import os, psutil

CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

def contract_random(contraction, d=2):
    inp, _ = contraction.split('->')
    sizes = inp.split(',')
    tensors = [np.random.randn(*[d]*len(s)) for s in sizes]
    print(f"Contracting {len(tensors)} tensors")
    r = np.einsum(contraction, *tensors)
    print('Done')
    return r


def test_mem(K, C=4, d=2, s=10):
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
    if K+C>len(CHARS):
        raise Exception('Too many indices')
    if C + np.ceil(K/d) > K:
        raise Exception('Too few indices')

    #-- Memory estimations
    elsize = np.random.randn(1).itemsize
    process = psutil.Process(os.getpid())
    overhead = process.memory_info().rss
    print('Overhead:', overhead)
    print('Should take:', elsize*2**K, 'Bytes')
    print('Total expected Memory:', overhead+elsize*2**K, 'Bytes')
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
        list(np.random.permutation(all_indices)[:s])
        for s in sm_sizes
    ]

    contraction = ','.join(
        ''.join(ix) for ix in dom_ix + sm_ix
    ) + '->' + ''.join(result_indices)

    print('contraction:', contraction)

    contract_random(contraction)

if __name__=="__main__":
    K = int(sys.argv[1])
    test_mem(K)

