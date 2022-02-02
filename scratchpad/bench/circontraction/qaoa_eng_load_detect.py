from functools import lru_cache
from matplotlib import backend_bases
import numpy as np
import networkx as nx
import platform
import pyrofiler
import qtensor
from qtensor import QtreeQAOAComposer
from qtensor import QAOAQtreeSimulator
from qtensor import toolbox
from qtensor.contraction_backends import get_mixed_backend
import pyrofiler.c as c
import pandas as pd
from functools import reduce


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
    for circ  in circuits:
        tn = qtensor.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(circ)
        opt = qtensor.toolbox.get_ordering_algo(algo)
        peo, _ = opt.optimize(tn)
        peos.append(peo)
    return peos

def gen_circ_peo_report_mix(circuit, peo, backend_name, threshold):

    curr_backend = get_mixed_backend(backend_name[0], backend_name[1], threshold)
    curr_sim = QAOAQtreeSimulator(QtreeQAOAComposer,backend=curr_backend)
    curr_sim.simulate_batch(circuit, peo = peo)


paramtest = [
    [20,4,3,"random"]
]

def process_one_series(result):
    first = result[0][0]
    processed = []
    for stamp in result:
        processed.append([stamp[0] - first, stamp[1]])
    return processed

def raw2df(rawTS, samplePeriod):
    df = pd.DataFrame(rawTS, columns = ["second", "cpu_util"])
    df['second'] = pd.to_datetime(df['second'], unit='s')
    df = df.set_index('second')
    df.index.name = None

    # Resampling and Interpo
    df1 = df.resample(samplePeriod).mean()
    df1.index = df1.index.strftime('%M:%S.%f')
    return df1

def df2reduced(collections):
    return reduce(lambda x,y: pd.concat((x,y), axis = 1).mean(axis = 1), collections)





if __name__ == '__main__':
    backends = [['einsum','cupy']]
    threshold = [16]
    my_reap = 3
    my_algo = 'greedy'
    
    for thr in threshold:

        for pb in paramtest:
            n, p, d, ttype = pb
            G, gamma, beta = get_test_problem(n,p,d,ttype)
            circs = gen_QAOA_circs(G, gamma, beta)
            peos = gen_circs_peos(circs,my_algo)

            for be in backends:
                cpus = []
                gpus = []
                for _ in range(my_reap):
                    with c.cpu_util_hist() as c_hist:
                        with c.gpu_util_hist() as g_hist:
                            for i, pack in enumerate(zip(circs, peos)):
                                circ, peo = pack
                                gen_circ_peo_report_mix(circ, peo, be, thr)
                    raw_cpu = process_one_series(c_hist.result)
                    cooked_cpu = raw2df(raw_cpu, '500ms')
                    cpus.append(cooked_cpu)

                    raw_gpu = process_one_series(g_hist.result)
                    cooked_gpu = raw2df(raw_gpu, '500ms')
                    gpus.append(cooked_gpu)
                reduced_cpu = df2reduced(cpus)
                reduced_gpu = df2reduced(gpus)
                print(reduced_cpu)
                print()
                print(reduced_gpu)