import argparse
    pche__        run_qtensor_qaoa_evals_gpu.py

    rint(f"Best string: {best_solution} with cut: {-best_cut}")
    return best_cut, circ
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
from utils import get_graphs, get_adj_mat
from qiskit_qaoa import get_maxcut_qaoa_ckt
from collections import defaultdict
from operator import itemgetter
from functools import partial


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



def compute_energy_graph(G, p, args):
    init_pt = np.random.uniform(0, 1, size=(2*p))
    backend = Aer.get_backend('qasm_simulator')
    obj = qaoa_obj(G, p, args)
    result = minimize(obj, init_pt, method='COBYLA', options={'maxiter':2500, 'disp': True})
    optimal = result['x']
    qc_opt = get_maxcut_qaoa_ckt(G, optimal[:p], optimal[p:], args)
    qc_opt.measure_all()
    counts = invert_counts(execute(qc_opt, backend).result().get_counts())
    best_cut, best_solution = min([(max_cut_obj(x,G),x) for x in counts.keys()], key=itemgetter(0))
    print(f"Best string: {best_solution} with cut: {-best_cut}")
    return best_cut, qc_opt


def par_fn(G, p, args):
    return compute_energy_graph(G, p,args)

# TODO: add way to get circuit from population
def main(args):
    graphs, energies = get_graphs('qtensor/qnas/qiskit_qnas/QAOA_Dataset/20_10_node_erdos_renyi_graphs.txt',
            'qtensor/qnas/qiskit_qnas/QAOA_Dataset/20_10_node_erdos_renyi_graphs_energies.txt')
    
    mixer_layers = ['x', 'xx', 'y', 'yy']
    
    result = []
    p = args.p
    for i, g in enumerate(graphs):
        print("running graph: ", i)
        fn = partial(par_fn,g, p)
        with mp.Pool(args.jobs) as pool:
            res = pool.starmap_async(par_fn, [(g, p, args) for _ in range(1)]) # generate a population of 150
            comp = res.get()
            best_energy, best_model = sorted([c for c in comp], key=lambda x: x[0])[0]
            result.append((best_energy, best_model))

    return result 
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, default=1, help='Depth of ansatz')
    parser.add_argument('-j', '--jobs', default=5, type=int, help='number of parallel jobs')

    args = parser.parse_args()
    args.nrots = 2
    args.nents = 2
    args.reps = 2
    os.makedirs('checkpoints/', exist_ok=True)
    result = main(args) # graphs
    for i in range(len(result)):
        with open(f'checkpoints/graph_model{i}.qpy', 'wb') as ofile:
            qpy.dump(result[i][1], ofile)
        
    best_energies = {i: e[0] for i, e in enumerate(result)}
    res_df = pd.DataFrame(best_energies, columns=['graph_idx', 'energies'])
    res_df.to_pickle(f'checkpoints/best_energies_graph_{args.p}.pkl')
    
    



    # res_df = pd.DataFrame(result)
    # res_df.to_pickle(f"graph_qaoa_p_{args.p}.pkl")
    





