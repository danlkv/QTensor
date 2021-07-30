import json
from functools import lru_cache
import numpy as np
import networkx as nx
import platform
import pyrofiler
import qtensor
from qtensor import QtreeQAOAComposer
from qtensor import QAOAQtreeSimulator
from qtensor.contraction_backends import get_backend


timing = pyrofiler.timing


@lru_cache
def get_test_problem(n=10, p=2, d=3, type='random'):
    #print('Test problem: n, p, d', n, p, d)
    if type == 'random':
        seed = 250
        G = nx.random_regular_graph(d, n, seed = 250)
    elif type == 'grid2d':
        G = nx.grid_2d_graph(n,n)
    elif type == 'line':
        G = nx.Graph()
        G.add_edges_from(zip(range(n-1), range(1, n)))
    gamma, beta = [np.pi/5]*p, [np.pi/2]*p
    return G, gamma, beta

def obj2dict(obj):
    keys = [x for x in dir(obj) if x[0]!='_']
    return dict((key, obj.__getattribute__(key)) for key in keys)


@lru_cache
def get_gpu_props_json():
    try:
        import torch
        devprops = torch.cuda.get_device_properties(torch.cuda.current_device())
        return obj2dict(devprops)
    except:
        return None

paramtest = [
    [24,4,3,"random"]
    ,[4, 4, 3, 'random']
    ,[10, 5, 2, 'random']
    ,[14, 1, 3, 'random']
    ,[3, 3, 0, 'grid2d']
    ,[8, 4, 0, 'line']
]

def mean_mmax(x: list):
    mx, mn = max(x), min(x)
    x.remove(mx)
    x.remove(mn)
    return np.mean(x)

def format_flops(flops):
    ord = 3*int(np.log10(flops)/3)
    suffix = {
        3: 'k'
        ,6: 'M'
        ,9: 'G'
        , 12: 'T'
    }[ord]
    return f'{(flops/10**ord).round(2)}{suffix}'


'''
Purpose: For every test problem, generate one fixed contraction peos
'''
def get_fixed_peos_for_a_pb(G, gamma, beta, algo:str, sim):
    peos, widths = [], []
    for edge in G.edges:
        cir = sim._edge_energy_circuit(G, gamma, beta, edge)
        tn = qtensor.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(cir)
        opt = qtensor.toolbox.get_ordering_algo(algo)
        peo, _ = opt.optimize(tn)
        width = opt.treewidth
        peos.append(peo)
        widths.append(width)
    return peos, widths



'''
Function: Simulate one lightcone/edge for one time, we dont look for a lightcone aggregation
CHANGE: Rid aggregation methods
'''
def gen_be_lc_report(G, gamma, beta, edge, peo, backend_name, gen_base = 0):

    curr_backend = get_backend(backend_name)
    curr_sim = QAOAQtreeSimulator(QtreeQAOAComposer,backend=curr_backend)
    circuit = curr_sim._edge_energy_circuit(G, gamma, beta, edge)
    curr_sim.simulate_batch(circuit, peo = peo)
    #curr_sim.backend.gen_report(show = False)



if __name__ == '__main__':
    gen_sim = QAOAQtreeSimulator(QtreeQAOAComposer)
    backends = ["cupy","torch_gpu","einsum","torch","tr_einsum","opt_einsum"]
    my_algo = 'rgreedy_0.05_30'
    for pb in [paramtest[0]]:

        '''
        Generate fixed peos for a given problem, thus be used for various backends
        '''
        with timing(callback=lambda x: None) as gen_pb:
            n, p, d, ttype = pb
            G, gamma, beta = get_test_problem(n=n,p=p,d=d, type = ttype)
            peos, widths = get_fixed_peos_for_a_pb(G, gamma, beta, algo = my_algo, sim = gen_sim)

        gen_base = gen_pb.result
        agg_reports = []
        for be in [backends[5]]:
            all_lightcones_report = []
            for i, pack in enumerate(zip(G.edges, peos)):
                edge, peo = pack
                for _ in range(3):
                    gen_be_lc_report(G, gamma, beta, edge, peo, be)
