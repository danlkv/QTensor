from qtensor.tools.lazy_import import quimb as qu
import networkx as nx
from itertools import repeat
import click

from qtensor.tools.lazy_import import cotengra as ctg

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
    circ = qu.tensor.circ_qaoa(terms, p, gammas, betas)
    args = list(zip(repeat(circ), repeat(kwargs), G.edges))

    with Pool(processes=n_processes) as pool:
        contributions = list(tqdm(pool.imap(edge_simulate, args), total=len(args)))
    return sum(contributions)


def simulate_one(G, p, **kwargs):
    terms = {(i, j):1 for i, j in G.edges}
    gammas, betas = [0.1]*p, [.2]*p
    circ = qu.tensor.circ_qaoa(terms, p, gammas, betas)

    contributions = []
    for edge in tqdm(G.edges):
        contributions.append(edge_simulate((circ, kwargs, edge)))
    dry_run = kwargs.get('dry_run', False)
    if dry_run:
        return contributions
    else:
        return sum(contributions)


def get_lightcone_tn(self, 
        G,
        where,
        simplify_sequence='ADCRS',
        simplify_atol=1e-12,
        dtype='complex128',
    ):
        r"""Compute the a single expectation value of operator ``G``, acting on
        sites ``where``, making use of reverse lightcone cancellation.

        .. math::

            \langle \psi_{\bar{q}} | G_{\bar{q}} | \psi_{\bar{q}} \rangle

        where :math:`\bar{q}` is the set of qubits :math:`G` acts one and
        :math:`\psi_{\bar{q}}` is the circuit wavefunction only with gates in
        the causal cone of this set. If you supply a tuple or list of gates
        then the expectations will be computed simulteneously.

        Parameters
        ----------
        self: quimb.tensor.Circuit
        G : array or tuple[array] or list[array]
            The raw operator(s) to find the expectation of.
        where : sequence of int
            Which qubits the operator acts on.
        optimize : str, optional
            Contraction path optimizer to use for the local expectation,
            can be a custom path optimizer as only called once (though path
            won't be cached for later use in that case).
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``opt_einsum``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        target_size : None or int, optional
            The largest size of tensor to allow. If specified and any
            contraction involves tensors bigger than this, 'slice' the
            contraction into independent parts and sum them individually.
            Requires ``cotengra`` currently.
        gate_opts : None or dict_like
            Options to use when applying ``G`` to the wavefunction.
        rehearse : bool, optional
            If ``True``, generate and cache the simplified tensor network and
            contraction path but don't actually perform the contraction.
            Returns a dict with keys ``'tn'`` and ``'info'`` with the tensor
            network that will be contracted and the corresponding contraction
            path if so.

        Returns
        -------
        scalar, tuple[scalar] or dict
        """

        rho = self.get_rdm_lightcone_simplified(
            where, simplify_sequence, simplify_atol
        )
        k_inds = tuple(self.ket_site_ind(i) for i in where)
        b_inds = tuple(self.bra_site_ind(i) for i in where)

        if isinstance(G, (list, tuple)):
            # if we have multiple expectations create an extra indexed stack
            nG = len(G)
            G_data = qu.tensor.tensor_core.do('stack', G, like=G[0])
            G_data = qu.tensor.tensor_core.reshape(G_data, (nG,) + (2,) * 2 * len(where))
            output_inds = (qu.tensor.tensor_core.rand_uuid(),)
        else:
            G_data = qu.tensor.tensor_core.reshape(G, (2,) * 2 * len(where))
            output_inds = ()

        TG = qu.tensor.tensor_core.Tensor(data=G_data, inds=output_inds + b_inds + k_inds)

        rhoG = rho | TG

        rhoG.full_simplify_(
            seq=simplify_sequence,
            atol=simplify_atol,
            output_inds=output_inds,
        )
        rhoG.astype_(dtype)
        return rhoG

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
    print(f"Time for N={N}, p={p}; E={E}: {end-start}")


if __name__=='__main__':
    bench_quimb()
