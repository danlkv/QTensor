from functools import lru_cache
import json
import numpy as np
import networkx as nx
import qtensor
from qtensor import QtreeQAOAComposer
from qtensor import QAOAQtreeSimulator
from qtensor.contraction_backends import get_backend, get_mixed_backend, get_embedded_backend
import pyrofiler
from pyrofiler import Profiler
from pyrofiler import callbacks
import random
from collections import defaultdict

random.seed(42)
np.random.seed(42)

@lru_cache
def get_test_problem(n=10, p=2, d=3, type='random'):
    print('Test problem: n, p, d', n, p, d)
    if type == 'random':
        G = nx.random_regular_graph(d, n, seed = 250)
    elif type == 'grid2d':
        G = nx.grid_2d_graph(n,n)
    elif type == 'line':
        G = nx.Graph()
        G.add_edges_from(zip(range(n-1), range(1, n)))
    gamma, beta = [np.pi/5]*p, [np.pi/2]*p
    return G, gamma, beta

def gen_QAOA_circs(G, gamma, beta):
    gen_sim = QAOAQtreeSimulator(QtreeQAOAComposer)
    circs = []
    for edge in G.edges:
        circ = gen_sim._edge_energy_circuit(G, gamma, beta, edge)
        circs.append(circ)
    return circs


def gen_circs_peos(circuits, algo):
    peos = []
    widths = []
    for circ  in circuits:
        tn = qtensor.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(circ)
        opt = qtensor.toolbox.get_ordering_algo(algo)
        peo, _ = opt.optimize(tn)
        treeWidth = opt.treewidth
        peos.append(peo)
        widths.append(treeWidth)
    return peos, widths

'''
TODO: a function to find the max peos and circ
'''
def get_max_peo_n_circ(circuits, peos, widths):
    max_width = max(widths)
    print("Max Tree Width: {}".format(max_width))
    max_index = widths.index(max_width)
    return circuits[max_index], peos[max_index]


def gen_circ_peo_report_mix(circuit, peo, backend_name, threshold):

    curr_backend = get_mixed_backend(backend_name[0], backend_name[1], threshold)
    curr_sim = QAOAQtreeSimulator(QtreeQAOAComposer,backend=curr_backend)
    curr_sim.simulate_batch(circuit, peo = peo)

def gen_circ_peo_report(circuit, peo, backend):
    curr_backend = get_backend(backend)
    curr_sim = QAOAQtreeSimulator(QtreeQAOAComposer,backend=curr_backend)
    curr_sim.simulate_batch(circuit, peo = peo)

def gen_circ_peo_report_embedded(circuit, peo, backend):
    curr_backend = get_embedded_backend(backend)
    curr_sim = QAOAQtreeSimulator(QtreeQAOAComposer,backend=curr_backend)
    curr_sim.simulate_batch(circuit, peo = peo)

    #Bucket Elimination is the loop
    #Create wrapper functions so that identifiers can be seen inside the profiling




paramtest = [
    [12,4,3,"random"]
]


class MyProfiler(Profiler):
    def __init__(self, callback=None):
        super().__init__(callback=callback)
        self.use_append()

    def get_stats(self, label):
        data = [x['value'] for x in self.data[label]]
        return dict(
            mean=np.mean(data),
            max = np.max(data),
            std = np.std(data),
            min = np.min(data)
        )
    
    # Transform the table to be
    # ref fun1 fun2
    def get_refs(self):
        pass

prof = MyProfiler()

'''
1. Pass the profiler globally
'''
pyrofiler.PROF = prof

'''
2. Wrapping the calback function so that the data is stored in this format:
    data[desc] = dict
'''
default_cb = prof.get_callback() #returns self._callbeck, the default callback in this case
def my_callback(value, desc, reference=0):
    default_cb(dict(reference=reference, value=value), desc) #wrapping, data[desc] = dict()
prof.set_callback(my_callback)

def main():
    backends = ['torch_gpu']
    my_reap = 5
    my_algo = 'greedy'
    callbacks.disable_printing()
    

    for pb in paramtest:
        n, p, d, ttype = pb
        G, gamma, beta = get_test_problem(n,p,d,ttype)
        circs = gen_QAOA_circs(G, gamma, beta)
        peos, widths = gen_circs_peos(circs,my_algo)

        maxCirc, maxPeo = get_max_peo_n_circ(circs, peos, widths)
        for be in backends:
            for _ in range(my_reap):
                gen_circ_peo_report_embedded(maxCirc, maxPeo, be)

main()

def reorderData(data):
    
    result = defaultdict(dict)
    
    for function in data:
        func_dict = defaultdict(list)
        
        # func_dict = ref -> list(value)
        for ref_val in data[function]:
            reference = ref_val['reference']
            value = ref_val['value']
            func_dict[reference].append(value)
        
        # result => ref -> {func1:[] func2:[]}
        for ref in func_dict:
            times = func_dict[ref]
            result[ref][function] = times
    
    return result


def describeData(data):
    result = defaultdict(dict)
    for ref in data:
        funcs = data[ref]
        for func in funcs:
            times = funcs[func]
            func_desc = dict(
                mean=np.mean(times),
                max = np.max(times),
                std = np.std(times),
                min = np.min(times)
            )
            result[ref][func] = func_desc
    return result

described = describeData(reorderData(prof.data))




def totalReport(data):
    report = []
    for entry, functions in data.items():
        item = {}
        item["bucket"] = entry[0]
        item["tensor"] = entry[1]
        for func in functions:
            item[func] = functions[func]['mean']
        report.append(item)

    return report

for rep in totalReport(described):
    print(json.dumps(rep))
