import argparse
import os
import numpy as np
import networkx as nx
import multiprocessing as mp
import pandas as pd
from qiskit import Aer, execute
from qiskit.tools.visualization import plot_histogram
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit.algorithms.minimum_eigensolvers import SamplingVQE, NumPyMinimumEigensolver, QAOA
from qiskit.algorithms.optimizers import SPSA
from qiskit.utils import algorithm_globals
from qiskit.primitives import Sampler
from scipy.optimize import minimize
from qiskit_optimization.algorithms import MinimumEigenOptimizer


from qiskit import qpy
from utils import get_graphs, get_adj_mat, attach_qubit_names
from qiskit_qaoa import get_maxcut_qaoa_ckt
from collections import defaultdict
from operator import itemgetter
from functools import partial
from itertools import repeat

from qtree.operators import from_qiskit_circuit
from qtensor.Simulate import QtreeSimulator, NumpyBackend
from qtensor import QiskitQAOAComposer, QtreeQAOAComposer
from qtensor.tools.mpi import mpi_map
from qtensor.contraction_backends import CuPyBackend

def invert_counts(count_dict):
    return {k[::-1]:v for k, v in count_dict.items()}

def max_cut_obj(x, G):
    cut = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            # the edge is cut
            cut -= 1
    return cut

def compute_maxcut_energy(counts, G):
    energy = 0
    total_counts = 0
    for meas, meas_count in counts.items():
        obj_for_meas = max_cut_obj(meas, G)
        energy += obj_for_meas * meas_count
        total_counts += meas_count
    return energy / total_counts



def qaoa_obj(G, p, args):
    backend = Aer.get_backend('qasm_simulator')

    # args = argparse.Namespace()
    # args.n_rots = 2
    # args.n_ents = 2
    # args.n_reps = 2

    def fun(theta):
        beta= theta[:p]
        gamma = theta[p:]
        qc = get_maxcut_qaoa_ckt(G, beta, gamma, args)
        qc.measure_all()
        counts = execute(qc, backend).result().get_counts()
        return compute_maxcut_energy(invert_counts(counts), G)
    return fun



def compute_energy_graph(G, p, args, sim):
    init_pt = np.random.uniform(0, 1, size=(2*p))
    obj = qaoa_obj(G, p, args)
    result = minimize(obj, init_pt, method='COBYLA', options={'maxiter':2500, 'disp': False})
    optimal = result['x']
    composer = QtreeQAOAComposer(G, gamma = optimal[p:], beta = optimal[:p])
    composer.ansatz_state()
    circ = composer.circuit
    counts = attach_qubit_names(sim.simulate_batch(qc = circ, batch_vars = composer.n_qubits))
    # counts = attach_qubit_names(counts)
    best_cut, best_solution = min([(max_cut_obj(x,G),x) for x in counts.keys()], key=itemgetter(0))
    return best_cut, circ

def par_fn(G, p, args, sim):
    return compute_energy_graph(G, p, args, sim)

def mpi_parallel_unit(arggen):
    G, p, args, sim = arggen
    #print(f"G: {G}, p: {p}, args: {args}, sim: {sim}")
    return compute_energy_graph(G, p, args, sim)

def get_args(G, p, args, sim):
    num_jobs = args.num_workers * args.num_samples_per_job
    return list(zip(repeat(G, num_jobs), repeat(p, num_jobs), repeat(args, num_jobs), repeat(sim, num_jobs)))

    # if jobs % num_workers == 0:
    #     # return arggen list of length jobs, each of which has jobs_per_worker number of 
    # else:
    #     first_set_of_jobs = num_circs - (total_jobs * num_circs_per_job)
    #     second_set_of_jobs = total_jobs - first_set_of_jobs


# TODO: add way to get circuit from population
def main(args):
    graphs, energies = get_graphs('/home/wberquis/repos/QTensor/qtensor/qnas/qiskit_qnas/QAOA_Dataset/20_10_node_erdos_renyi_graphs.txt',
            '/home/wberquis/repos/QTensor/qtensor/qnas/qiskit_qnas/QAOA_Dataset/20_10_node_erdos_renyi_graphs_energies.txt')
    
    mixer_layers = ['x', 'xx', 'y', 'yy']
    
    sim = QtreeSimulator(backend=CuPyBackend())
    results = []
    p = args.p
    for i, g in enumerate(graphs):
        print("running graph: ", i)
        arggen = get_args(g, p, args, sim)
        comp = mpi_map(mpi_parallel_unit, arggen, pbar = True, total = args.num_nodes)
        #print(len(comp), comp)
        if comp:
            best_energy, best_model = sorted([c for c in comp], key=lambda x: x[0])[0]
        # best_energy, best_model = compute_energy_graph(g, p, args, sim)
            results.append((best_energy, best_model))
    if results:
        return results
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, default=1, help='Depth of ansatz')
    parser.add_argument('-j', '--jobs', default=1, type=int, help='number of parallel jobs')
    parser.add_argument('-n', '--np', default=1, type=int, help='Total number of MPI ranks; number_nodes * ranks_per_node')
    parser.add_argument('-ppn', '--ppn', default=1, type=int, help='Number of MPI ranks per node')
    parser.add_argument('-d', '--depth', default=1, type=int, help='Number of hardware threads per rank, spacing between MPI ranks on a node')
    parser.add_argument('-s', '--samples', default=1, type=int, help='Number of samples each MPI worker will have')


    args = parser.parse_args()
    args.nrots = 2
    args.nents = 2
    args.reps = 2

    # add to parser
    args.num_nodes = 2
    num_gpus = 1
    num_cpus = 32
    num_threads = 1

    args.num_workers = args.num_nodes*num_gpus*num_cpus*num_threads
    args.num_samples_per_job = 10

    os.makedirs('checkpoints/', exist_ok=True)


    result = main(args) # graphs
    for i in range(len(result)):
        with open(f'checkpoints/graph_model{i}.qpy', 'wb') as ofile:
            qpy.dump(result[i][1], ofile)
        
    best_energies = {i: e[0] for i, e in enumerate(result)}
    res_df = pd.DataFrame(best_energies, columns=['graph_idx', 'energies'])
    res_df.to_pickle(f'checkpoints/best_energies_graph_{args.p}.pkl')
    print(result)
    



    # res_df = pd.DataFrame(result)
    # res_df.to_pickle(f"graph_qaoa_p_{args.p}.pkl")
    





