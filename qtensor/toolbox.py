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


def qaoa_energy_tw_from_graph(G, p, max_time=0, max_tw=0,
                              ordering_algo='greedy', print_stats=False):
    gamma, beta = [0]*p, [0]*p
    def get_tw(circ):

        tn = QtreeTensorNet.from_qtree_gates(circ)

        if ordering_algo=='greedy':
            opt = OrderingOptimizer()
        elif ordering_algo=='tamaki':
            opt = TamakiOptimizer(wait_time=45)
        elif ordering_algo=='without':
            opt = WithoutOptimizer()
        else:
            raise ValueError("Ordering algorithm not supported")
        peo, tn = opt.optimize(tn)
        treewidth = opt.treewidth
        return treewidth

    twidths = []
    if max_time:
        start = time.time()
    else:
        start = np.inf
    with tqdm(total=G.number_of_edges(), desc='Edge iteration') as pbar:
        for edge in G.edges():
            composer = QtreeQAOAComposer(G, beta=beta, gamma=gamma)
            composer.energy_expectation_lightcone(edge)
            tw = get_tw(composer.circuit)
            pbar.update()
            subgraph = get_edge_subgraph(G, edge, len(beta))
            pbar.set_postfix(current_tw=tw, subgraph_nodes=subgraph.number_of_nodes())
            if max_tw:
                if tw>max_tw:
                    print(f'Encountered treewidth of {tw}, which is larger {max_tw}')
                    break
            twidths.append(tw)
            if time.time() - start > max_time:
                break
    if print_stats:
        print(f'med={np.median(twidths)} mean={round(np.mean(twidths), 2)} max={np.max(twidths)}')
    return twidths
