import qtensor
import numpy as np
from qtensor.tools.lazy_import import torch
from functools import partial
from tqdm.auto import tqdm
from qtensor.contraction_backends import TorchBackend
import networkx as nx
from functools import lru_cache
from cartesian_explorer.caches import JobLibCache

cache = JobLibCache('/tmp/qtensor_bethe_order')

def qaoa_maxcut_torch(G, gamma, beta, edge=None,
                      **kwargs
                     ):
    """
    Find optimal parameters for qaoa max cut

    Args:
        G: graph for MaxCut
        gamma, beta: iterable of floats
        edge: optional edge of graph, if None,
            the sum over all edges is calculated
        steps (int): number of optimization steps
        Opt (torch.Optimizer): optimizer to use
        opt_kwargs (dict): args to optimizer

    Returns:
        loss_history: array of losses with len=steps
        param_history: array of (gamma, beta) with len=steps
    """
    params = [torch.tensor(x, requires_grad=True)
              for x in [gamma, beta]]

    ordering_algo = kwargs.pop('ordering_algo', 'greedy')
    if edge is not None:
        peo = _edge_peo(p=len(gamma), G=G, edge=edge, ordering_algo=ordering_algo)
    else:
        peo = _energy_peo(p=len(gamma), G=G, ordering_algo=ordering_algo)

    if edge is not None:
        loss = partial(_edge_loss, edge=edge, G=G, peo=peo)
    else:
        loss = partial(_energy_loss, G=G, peos=peo)
    params = [torch.tensor(x, requires_grad=True)
              for x in [gamma, beta]]

    return optimize_params(loss, *params, **kwargs)


def optimize_params(get_loss, *params, steps=50,
                    pbar=True,
                    Opt=torch.optim.RMSprop,
                    opt_kwargs=dict(),
                    **kwargs):

    opt = Opt(params=params, **opt_kwargs)
    losses = []
    param_history = []
    param_history.append([x.detach().numpy().copy() for x in params])

    if pbar:
        _pbar = tqdm(total=steps)
    else:
        _pbar = None

    for i in range(steps):
        loss = get_loss(*params, **kwargs)
        opt.zero_grad()

        loss.backward()
        opt.step()

        losses.append(loss.detach().numpy().data)
        param_history.append([x.detach().numpy().copy() for x in params])
        if pbar:
            _pbar.update(1)
    loss = get_loss(*params, **kwargs)
    losses.append(loss.detach().numpy())
    losses = np.array(losses).flatten()
    return losses, param_history

@cache
#@lru_cache
def _edge_peo(p, G, edge, ordering_algo):
    opt = qtensor.toolbox.get_ordering_algo(ordering_algo)
    composer = qtensor.DefaultQAOAComposer(G, gamma=[0.1]*p, beta=[.3]*p)
    composer.energy_expectation_lightcone(edge)

    tn = qtensor.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(composer.circuit)
    peo, _ = opt.optimize(tn)
    print('Treewidth', opt.treewidth)
    return peo

@cache
#@lru_cache
def _energy_peo(p, G, ordering_algo):
    opt = qtensor.toolbox.get_ordering_algo(ordering_algo)
    peos = []
    for edge in G.edges():
        composer = qtensor.DefaultQAOAComposer(G, gamma=[0.1]*p, beta=[.3]*p)
        composer.energy_expectation_lightcone(edge)
        tn = qtensor.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(composer.circuit)
        peo, _ = opt.optimize(tn)
        print('tw', opt.treewidth)
        peos.append(peo)
    return peos

def _energy_loss(gamma, beta, G, peos=None):
    backend = TorchBackend()
    sim = qtensor.QtreeSimulator(backend=backend)

    composer = qtensor.TorchQAOAComposer(G, gamma=gamma, beta=beta)
    loss = torch.tensor([0.])

    if peos is None:
        peos = [None]*G.number_of_edges()

    for edge, peo in zip(G.edges, peos):
        composer.energy_expectation_lightcone(edge)
        #print('peo', peo)
        loss += torch.real(sim.simulate_batch(composer.circuit, peo=peo))
        composer.builder.reset()
    return -(G.number_of_edges() - loss)/2


def _edge_loss(gamma, beta, G, edge, peo=None):
    backend = TorchBackend()
    sim = qtensor.QtreeSimulator(backend=backend)

    composer = qtensor.TorchQAOAComposer(G, gamma=gamma, beta=beta)
    composer.energy_expectation_lightcone(edge)
    loss = torch.real(sim.simulate_batch(composer.circuit, peo=peo))
    return loss

def evaluate_edge_energy_with_grad(G, gamma, beta, edge, peo=None, ordering_algo='greedy'):
    """
    Args:
        G (nx.Graph): QAOA MaxCut graph
        gamma (np.array): gamma params for QAOA
        beta (np.array): beta params for QAOA
        edge (tuple): Edge from G
        peo (list): Elimination order, if None (default) then use ``ordering_algo`` to find one
        ordering_algo (str): ordering algorithm to use if ``peo`` is None

    Returns:
        tuple(loss, gamma_grad, beta_grad)
    """
    params = [torch.tensor(x, requires_grad=True)
              for x in [gamma, beta]]

    peo = _edge_peo(p=len(gamma), G=G, edge=edge, ordering_algo=ordering_algo)

    get_loss = partial(_edge_loss, edge=edge, G=G, peo=peo)

    loss = get_loss(*params)
    loss.backward()

    return loss.detach().numpy(), params[0].grad.detach().numpy(), params[1].grad.detach().numpy()

