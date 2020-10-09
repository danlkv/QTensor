import networkx as nx
import numpy as np
from tqdm.auto import tqdm
import time

from qtensor.optimisation.TensorNet import QtreeTensorNet
from qtensor.optimisation.Optimizer import OrderingOptimizer, TamakiOptimizer, WithoutOptimizer
from qtensor.utils import get_edge_subgraph
from qtensor import QtreeQAOAComposer

def random_graph(nodes, type='random', **kwargs):
    """
    Generate a random graph
    Parameters:
        nodes: int
            Number of nodes in the graph
        type: enum['random', 'erdos_renyi']
            algorithm to use
        **kwargs:
            keyword arguments to specific algorithm
            usually of form
                seed: int
                degree: int
    """
    if type == 'random':
        return nx.random_regular_graph(n=nodes, d=kwargs['degree']
                                 , seed=kwargs.get('seed'))
    if type == 'erdos_renyi':
        prob_of_edge_add = kwargs['degree']/(nodes-1)
        return nx.erdos_renyi_graph(n=nodes, p=prob_of_edge_add
                                    , seed=kwargs.get('seed'))
    else:
        raise ValueError('Unsupported graph type')



def optimize_circuit(circ, algo='greedy'):

    if algo=='greedy':
        opt = OrderingOptimizer()
    elif algo=='tamaki':
        opt = TamakiOptimizer(wait_time=45)
    elif algo=='without':
        opt = WithoutOptimizer()
    else:
        raise ValueError("Ordering algorithm not supported")

    tn = QtreeTensorNet.from_qtree_gates(circ)
    peo, tn = opt.optimize(tn)
    return peo, tn, opt

def get_tw(circ, ordering_algo='greedy'):
    peo, tn, opt = optimize_circuit(circ, algo=ordering_algo)
    treewidth = opt.treewidth
    return treewidth

def get_cost_params(circ, ordering_algo='greedy', overflow_tw=None):
    peo, tn, opt = optimize_circuit(circ, algo=ordering_algo)
    treewidth = opt.treewidth
    print('tw', treewidth)
    if overflow_tw is not None:
        if treewidth > overflow_tw:
            mems, flops = [np.inf], [np.inf]
            return treewidth, np.inf, np.inf
    mems, flops = tn.simulation_cost(peo)
    return treewidth, max(mems), sum(flops)


def qaoa_energy_lightcone_iterator(G, p, max_time=None):
    gamma, beta = [0.1]*p, [0.3]*p
    if max_time:
        start = time.time()
    else:
        start = np.inf
    for edge in G.edges():
        composer = QtreeQAOAComposer(G, beta=beta, gamma=gamma)
        composer.energy_expectation_lightcone(edge)
        subgraph = get_edge_subgraph(G, edge, len(beta))
        yield composer.circuit, subgraph
        if time.time() - start > max_time:
            break


def qaoa_energy_cost_params_stats_from_graph(G, p, max_time=0, max_tw=None,
                              ordering_algo='greedy', print_stats=False):
    cost_params = []
    tw = mem = flop = 0
    with tqdm(total=G.number_of_edges(), desc='Edge iteration') as pbar:
        for circ, subgraph in qaoa_energy_lightcone_iterator(G, p, max_time=max_time):
            _tw, _m, _f = cost_params(circ, ordering_algo=ordering_algo, overflow_tw=max_tw)
            tw = max(tw, _tw)
            mem = max(mem, _m)
            flop += _f
            pbar.set_postfix(current_tw=tw, subgraph_nodes=subgraph.number_of_nodes())
    return tw, mem, flop


def qaoa_energy_tw_from_graph(G, p, max_time=0, max_tw=0,
                              ordering_algo='greedy', print_stats=False):
    twidths = []
    with tqdm(total=G.number_of_edges(), desc='Edge iteration') as pbar:
        for circuit, subgraph in qaoa_energy_lightcone_iterator(G, p, max_time=max_time):
            tw = get_tw(circuit, ordering_algo=ordering_algo)
            pbar.update()
            pbar.set_postfix(current_tw=tw, subgraph_nodes=subgraph.number_of_nodes())
            if max_tw:
                if tw>max_tw:
                    print(f'Encountered treewidth of {tw}, which is larger {max_tw}')
                    break
            twidths.append(tw)

    if print_stats:
        print(f'med={np.median(twidths)} mean={round(np.mean(twidths), 2)} max={np.max(twidths)}')
    return twidths


def qaoa_energy_cost_params_from_graph(G, p, max_time=0, max_tw=0,
                              ordering_algo='greedy', print_stats=False):
    costs = []
    with tqdm(total=G.number_of_edges(), desc='Edge iteration') as pbar:
        for circuit, subgraph in qaoa_energy_lightcone_iterator(G, p, max_time=max_time):
            c = get_cost_params(circuit, ordering_algo=ordering_algo)
            if max_tw:
                if c[0]>max_tw:
                    print(f'Encountered treewidth of {c[0]}, which is larger {max_tw}')
                    break
            costs.append(c)

            pbar.update(1)
            pbar.set_postfix(current_costs=c, subgraph_nodes=subgraph.number_of_nodes())
    if print_stats:
        print(f'med={np.median(costs)} mean={round(np.mean(costs), 2)} max={np.max(costs)}')
    return costs
