import quimb as qu
import quimb.tensor as qtn
import networkx as nx
from itertools import repeat
import click

import cotengra as ctg

from multiprocessing import Pool

import time
from tqdm import tqdm
SEED = 100

def edge_simulate(args):
    G, p, kwargs, edge = args

    terms = {(i, j):1 for i, j in G.edges}
    gammas, betas = [0.1]*p, [.2]*p
    circ = qtn.circ_qaoa(terms, p, gammas, betas)
    ZZ = qu.pauli('Z') & qu.pauli('Z')
    optimizer = ctg.HyperOptimizer(
        parallel=False,
        max_repeats=10000,
        max_time=kwargs.get('optimizer_time', 1)
    )
    #return circ.local_expectation(ZZ, edge, optimize=optimizer)
    return circ.local_expectation(ZZ, edge, optimize=optimizer)

def simulate_one_parallel(N, p, n_processes=28, **kwargs):
    G = nx.random_regular_graph(3, N, seed=SEED)
    args = list(zip(repeat(G), repeat(p), repeat(kwargs), G.edges))

    with Pool(processes=n_processes) as pool:
        contributions = list(tqdm(pool.imap(edge_simulate, args), total=len(args)))
    return sum(contributions)

def simulate_one(N, p):
    G = nx.random_regular_graph(3, N, seed=SEED)

    terms = {(i, j):1 for i, j in G.edges}
    gammas, betas = [0.1]*p, [.2]*p
    circ = qtn.circ_qaoa(terms, p, gammas, betas)
    ZZ = qu.pauli('Z') & qu.pauli('Z')

    contributions = []
    for edge in tqdm(G.edges):
        contributions.append(circ.local_expectation(ZZ, edge))
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
    start = time.time()
    #E = simulate_one(N, p)
    E = simulate_one_parallel(N, p, n_processes=n_processes, optimizer_time=optimizer_time)
    end = time.time()
    print(f'Time for {N=}, {p=}; {E=}: {end-start}')

if __name__=='__main__':
    bench_quimb()
