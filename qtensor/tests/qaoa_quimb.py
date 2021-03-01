import quimb as qu
import quimb.tensor as qtn
import networkx as nx
from itertools import repeat
import click

import cotengra as ctg

from multiprocessing import Pool

import time
from tqdm.auto import tqdm
SEED = 100

def edge_simulate(args):
    circ, kwargs, edge = args

    ZZ = qu.pauli('Z') & qu.pauli('Z')
    opt_type = kwargs.get('ordering_algo', 'uniform')
    max_repeats = kwargs.get('max_repeats', 10)
    if opt_type == 'hyper':
        optimizer = ctg.HyperOptimizer(
            parallel=False,
            max_repeats=max_repeats,
            max_time=kwargs.get('optimizer_time', 1)
        )
    elif opt_type == 'uniform':
        optimizer = ctg.UniformOptimizer(
            parallel=False,
            methods=['greedy'],
            max_repeats=max_repeats,
            max_time=kwargs.get('optimizer_time', 1)
        )
    else:
        raise ValueError('Ordering algorithm not supported')
    #return circ.local_expectation(ZZ, edge, optimize=optimizer)
    simplify_sequence = kwargs.get('simplify_sequence', 'ADCRS')
    dry_run = kwargs.get('dry_run', False)
    if dry_run:
        return circ.local_expectation_rehearse(ZZ, edge, optimize=optimizer, simplify_sequence=simplify_sequence)
    else:
        return circ.local_expectation(ZZ, edge, optimize=optimizer, simplify_sequence=simplify_sequence)

def simulate_one_parallel(G, p, n_processes=28, **kwargs):
    terms = {(i, j):1 for i, j in G.edges}
    gammas, betas = [0.1]*p, [.2]*p
    circ = qtn.circ_qaoa(terms, p, gammas, betas)
    args = list(zip(repeat(circ), repeat(kwargs), G.edges))

    with Pool(processes=n_processes) as pool:
        contributions = list(tqdm(pool.imap(edge_simulate, args), total=len(args)))
    return sum(contributions)

def simulate_one(G, p, **kwargs):
    terms = {(i, j):1 for i, j in G.edges}
    gammas, betas = [0.1]*p, [.2]*p
    circ = qtn.circ_qaoa(terms, p, gammas, betas)

    contributions = []
    for edge in tqdm(G.edges):
        contributions.append(edge_simulate((circ, kwargs, edge)))
    dry_run = kwargs.get('dry_run', False)
    if dry_run:
        return contributions
    else:
        return sum(contributions)

@click.command()
@click.option('-n', '--nodes', default=100)
@click.option('-p', default=2)
@click.option('-P', '--n-processes', default=1)
@click.option('-S', '--seed', default=10)
@click.option('-T', '--optimizer-time', default=1.)
def bench_quimb(nodes, p, n_processes, seed=10, optimizer_time=1):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    N = nodes
    G = nx.random_regular_graph(3, N, seed=SEED)
    start = time.time()
    if n_processes==1:
        E = simulate_one(G, p, optimizer_time=optimizer_time)
    else:
        E = simulate_one_parallel(G, p, n_processes=n_processes, optimizer_time=optimizer_time)
    end = time.time()
    print(f'Time for {N=}, {p=}; {E=}: {end-start}')

if __name__=='__main__':
    bench_quimb()
