import numpy as np
import random
from itertools import accumulate
import pyrofiler
import opt_einsum
import sys
import os, psutil
import fire
try:
    import ctf
except ImportError:
    print('Can\'t import ctf')

ELEMENT_SIZE = np.random.randn(1).itemsize

CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

def tcontract(contraction, *tensors, optimize='random-greedy'):
    inp, out = contraction.split('->')
    ixs = inp.split(',')
    if len(ixs) > 2:
        raise Exception('Not supported more than 2!')
    # --
    # \sum_k A_{kfm} * B_{kfn} = C_{fmn}
    common = set(ixs[0]).intersection(set(ixs[1]))
    mix = set(ixs[0]) - common
    nix = set(ixs[1]) - common
    kix = common - set(out)
    fix = common - kix
    common = list(kix) + list(fix)
    print('t0', tensors[0].shape, len(tensors[0].shape))
    print('common' , common)
    print('mix' , mix)

    a = tensors[0].transpose(*[
        list(ixs[0]).index(x) for x in common + list(mix)
    ])

    b = tensors[1].transpose(*[
        list(ixs[1]).index(x) for x in common + list(nix)
    ])

    k, f, m, n = 2**len(kix), 2**len(fix), 2**len(mix), 2**len(nix)
    a = a.reshape(k, f, m)
    b = b.reshape(k, f, n)

    c = np.empty((f, m, n), dtype=np.float64)

    import tcontract
    with pyrofiler.timing('contraction time'):
        tcontract.debug_mkl_contract_sum_float(a, b, c)
    print('> mkl first elem', c.ravel()[0])
    with pyrofiler.timing('contraction time eins'):
        G = np.einsum('ijk,ijl->jkl', a, b)
    print('> eins first elem', G.ravel()[0])

    return c

def opt_einsum_checking(contraction, *tensors, optimize='random-greedy'):
    uniq_ids = set(contraction) - {',', '-', '>'}
    sizes_dict = {i:2 for i in uniq_ids}
    views = opt_einsum.helpers.build_views(contraction, sizes_dict)

    with pyrofiler.timing('optimization time'):
        path, info = opt_einsum.contract_path(contraction, *views, optimize=optimize)

    print('Largest memory', ELEMENT_SIZE*info.largest_intermediate)
    print('GFlops', float(info.opt_cost)/1e9)
    print(info)
    expect = len(contraction.split('->')[1])
    if 2**expect < info.largest_intermediate:
        print(f'WARN: Optimizer {optimize} did not do a good job: '
              f'expected {2**expect} got {info.largest_intermediate} for contraction')
    return opt_einsum.contract(contraction, *tensors, optimize=path)

def exatn_contract(contraction, *tensors):
    import sys
    sys.path.append('/home/plate/.local/lib')
    import exatn

    inp, out = contraction.split('->')
    ixs = inp.split(',')

    _r_name = 'R'
    exatn.createTensor(_r_name, [2]*len(out), 0.0)
    _t_names = [f'T{i}' for i in range(len(tensors))]
    print('tnames', _t_names)
    [exatn.createTensor(name, t.copy(order='F'))
     for name, t in zip(_t_names, tensors)
    ]

    _ix2exatn = lambda x: ','.join(x)
    _exatn_expr = f'R({_ix2exatn(out)})='
    _exatn_expr += '*'.join(f'{tname}({_ix2exatn(ix)})'
                            for tname, ix in zip(_t_names, ixs))
    print('exatn expr', _exatn_expr)
    #exatn.contractTensors(_exatn_expr, 1.0)
    exatn.evaluateTensorNetwork('ff', _exatn_expr)
    res = exatn.getLocalTensor(_r_name)
    return res.transpose()


def get_backend(backend):
    return {
        'opt_einsum': opt_einsum_checking,
        'einsum': np.einsum,
        'tcontract': tcontract,
        'exatn': exatn_contract,
        'ctf': lambda c, *t: ctf.core.einsum(c, *[ctf.astensor(x) for x in t]),
    }[backend]

@pyrofiler.mem_util('memory to contract')
def contract_random(contraction, dim=2, backend='opt_einsum'):
    inp, out = contraction.split('->')
    sizes = inp.split(',')
    print([len(x) for x in sizes], len(out))
    tensors = [np.random.randn(*[dim]*len(s)) for s in sizes]
    print(f"Contracting {len(tensors)} tensors")
    with pyrofiler.timing('time to contract:'):
        r = get_backend(backend)(contraction, *tensors)
    print('Done')
    print(r.ravel()[0])
    return r


def test_mem(K, C=4, d=2, s=10, backend='opt_einsum', seed=10, expr=None):
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
        list(np.random.permutation(all_indices)[:s])
        for s in sm_sizes
    ]

    contraction = ','.join(
        ''.join(ix) for ix in sm_ix + dom_ix
    ) + '->' + ''.join(result_indices)

    print('contraction:', contraction)

    contract_random(contraction, backend=backend)

if __name__=="__main__":
    fire.Fire(test_mem)

