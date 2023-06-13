import qtensor
import numpy as np
import networkx as nx
from qtensor.optimisation.Optimizer import SlicesOptimizer

from qtensor.tools.mpi import mpi_map, print_stats
from mpi4py import MPI

SEED = 1

class MPIPool:
    def __init__(self, nprocs):
        # nprocs is ignored, since we use MPI_COMM_WORLD
        self.nprocs = nprocs
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def map(self, callable, args):
        # currently, mpi_map helper only supports functions.
        # wrapper prevents passing methods or other non-function callables.
        def wrapper_fn(*args, **kwargs):
            return callable(*args, **kwargs)
        return mpi_map(wrapper_fn, args)

    def imap(self, func, args):
        if self.rank == 0:
            return self.map(func, args)
        else:
            self.map(func, args)
            return [0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def simulate_prob_mpi():
    nqubits = 30
    p = 3
    G = nx.random_regular_graph(3, nqubits, seed=SEED)
    gamma, beta = [np.pi/3]*p, [np.pi/2]*p

    composer = qtensor.QtreeQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()

    sim = qtensor.FeynmanSimulator(
        optimizer=SlicesOptimizer(max_tw=25, tw_bias=0)
        , pool_type='thread'
    )
    sim.pool = MPIPool
    result = sim.simulate(composer.circuit)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Result:", result)
        print("="*80)
        print_stats()

if __name__ == "__main__":
    simulate_prob_mpi()
