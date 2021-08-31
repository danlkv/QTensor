import json
from functools import lru_cache
import numpy as np
import cupy as cp
import torch
import random
import networkx as nx
import platform
import pyrofiler
import qtensor
import time
from qtensor import QtreeQAOAComposer
from qtensor import QAOAQtreeSimulator
from qtensor.contraction_backends import get_mixed_backend, get_mixed_perf_backend, get_gpu_perf_backend, get_cpu_perf_backend

gpu_backends = ['torch_gpu', 'cupy', 'tr_torch', 'tr_cupy', 'tr_cutensor']
cpu_backends = ['einsum', 'torch_cpu', 'mkl', 'opt_einsum', 'tr_einsum', 'opt_einsum']

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

timing = pyrofiler.timing

random.seed(42)
np.random.seed(42)

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

paramtest = [
    # n, p, degree, type
     [12, 4, 3, 'random']
    ,[10, 5, 2, 'random']
    ,[14, 1, 3, 'random']
    ,[3, 3, 0, 'grid2d']
    ,[8, 4, 0, 'line']
]

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

def format_flops(flops):
    ord = 3*int(np.log10(flops)/3)
    suffix = {
        3: 'k'
        ,6: 'M'
        ,9: 'G'
        , 12: 'T'
    }[ord]
    return f'{(flops/10**ord).round(2)}{suffix}'

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

def mean_mmax(x: list):
    mx, mn = max(x), min(x)
    x.remove(mx)
    x.remove(mn)
    return np.mean(x)



'''
Function:   Simulate one lightcone for one time, only for mixed be
Attempt:    Only for Mixed Be
I/O: -> list, np.ndarray
'''
def gen_mixed_lc_report(G, gamma, beta, edge, peo, backend_name, gen_base = 0):

    curr_backend = get_mixed_perf_backend(backend_name[0], backend_name[1], backend_name[2])   
    curr_sim = QAOAQtreeSimulator(QtreeQAOAComposer,backend=curr_backend)
    circuit = curr_sim._edge_energy_circuit(G, gamma, beta, edge)
    curr_sim.simulate_batch(circuit, peo = peo)
    
    curr_backend.cpu_be.gen_report(show = False)
    curr_backend.gpu_be.gen_report(show = False)

    '''
    Merging two report table together
    CPU table on top, gpu second
    '''
    cpu_table = np.asarray(curr_backend.cpu_be.report_table.records)
    gpu_table = np.asarray(curr_backend.gpu_be.report_table.records)
    report_record = np.vstack((cpu_table, gpu_table))
    #print("CPU Table Shape: ",cpu_table.shape)
    #print(gpu_table.shape)
    #print(all_table.shape)

    '''
    Checking Tensor Dims/bytes
    1st Loop for cpu, 2nd loop for gpu
    '''
    tensor_dims = []
    for i,x in enumerate(curr_backend.cpu_be._profile_results):
        bc_signature, _ = curr_backend.cpu_be._profile_results[x]
        bucket_size = []
        for tensor in bc_signature:
            tensor_2_size = [qtreeVar.size for qtreeVar in tensor]
            tensor_dim = np.prod(tensor_2_size)
            bucket_size.append(tensor_dim)
        tensor_dims.append(sum(bucket_size))
    
    for i,x in enumerate(curr_backend.gpu_be._profile_results):
        bc_signature, _ = curr_backend.gpu_be._profile_results[x]
        bucket_size = []
        for tensor in bc_signature:
            tensor_2_size = [qtreeVar.size for qtreeVar in tensor]
            tensor_dim = np.prod(tensor_2_size)
            bucket_size.append(tensor_dim)
        tensor_dims.append(sum(bucket_size))
    
    '''
    Processing tensor_dims and gen_times as new columns
    '''
    tensor_dims = np.asarray(tensor_dims).reshape(-1,1)
    gen_time = gen_base
    gen_times = np.full((tensor_dims.shape), gen_time)


    '''
    Generate titles
    '''
    title_record = curr_backend.cpu_be.report_table._title_row()[1:]
    title_record.append("byte")
    title_record.append("gen_time")

    '''
    Modify new report table
    '''
    report_record = np.hstack((report_record,tensor_dims))
    report_record = np.hstack((report_record,gen_times))

    return title_record, report_record



'''
Function: Simulate one lightcone for one time, only for pure be
TODO: make sure the I/O are the same
'''
def gen_be_lc_report(G, gamma, beta, edge, peo, backend_name, gen_base = 0):

    if backend_name in gpu_backends:
        curr_backend = get_gpu_perf_backend(backend_name)
    else:
        curr_backend = get_cpu_perf_backend(backend_name)
    curr_sim = QAOAQtreeSimulator(QtreeQAOAComposer,backend=curr_backend)
    circuit = curr_sim._edge_energy_circuit(G, gamma, beta, edge)
    curr_sim.simulate_batch(circuit, peo = peo)
    curr_sim.backend.gen_report(show = False)
    report_record = np.asarray(curr_backend.report_table.records)
    #print(report_record.shape)

    '''
    Generate report table data
    '''
    tensor_dims = []
    for i, x in enumerate(curr_sim.backend._profile_results):
        bucket_signature, _ = curr_sim.backend._profile_results[x]
        bucket_size = []
        #print(len(max(bucket_signature, key=len)))
        for tensor in bucket_signature:
            tensor_2_size = [qtreeVar.size for qtreeVar in tensor]
            tensor_dim = np.prod(tensor_2_size)
            bucket_size.append(tensor_dim)
        tensor_dims.append(sum(bucket_size))

    tensor_dims = np.asarray(tensor_dims).reshape(-1,1)
    gen_time = gen_base
    gen_times = np.full((tensor_dims.shape), gen_time)

    title_record = curr_sim.backend.report_table._title_row()[1:]
    title_record.append("byte")
    title_record.append("gen_time")

    report_record = np.hstack((report_record,tensor_dims))
    report_record = np.hstack((report_record,gen_times))

    return title_record, report_record


def collect_reports_for_a_lc(G, gamma, beta, edge, peo, backend_name, repeat, gen_base):
    bucketindex_2_reports = {}
    '''
    Collect each lc's report for [repeat] times
    '''
    for _ in range(repeat):
        if type(backend_name) == list:
            titles, report_table = gen_mixed_lc_report(G, gamma, beta, edge, peo, backend_name, gen_base)
        else:
            titles, report_table = gen_be_lc_report(G, gamma, beta, edge, peo, backend_name, gen_base)
        for i, report in enumerate(report_table):
            if i not in bucketindex_2_reports:
                bucketindex_2_reports[i] = [report]
            else:
                bucketindex_2_reports[i].append(report)
    
    #print(bucketindex_2_reports)
    #
    return titles, bucketindex_2_reports


def reduce_bucket_reports(G, gamma, beta, edge, peo, backend_name, repeat, gen_base):
    bi_2_reduced = {}
    titles, bi_2_reports = collect_reports_for_a_lc(G, gamma, beta, edge, peo, backend_name, repeat, gen_base)
    
    for bi, report in bi_2_reports.items():
        bi_redux = []
        t_report = np.transpose(report)
        for attr in t_report:
            bi_redux.append(mean_mmax(attr.tolist()))
        bi_2_reduced[bi] = bi_redux
    return titles, bi_2_reduced

def process_reduced_data(G, gamma, beta, edge, peo, backend_name, problem, repeat, gen_base, lc_index, opt_algo):
    if type(backend_name) == list:
        final_backend_name = backend_name[0]+"-"+backend_name[1]
        threshold = backend_name[2]
    else:
        final_backend_name = backend_name
        threshold = 0
    titles, bi_2_reduced = reduce_bucket_reports(G, gamma, beta, edge, peo, backend_name, repeat, gen_base)
    GPU_PROPS = get_gpu_props_json()
    lc_collection = []
    for bi, report in bi_2_reduced.items():
        bi_json_usable = {}
        bi_json_usable["backend"] = final_backend_name
        bi_json_usable["threshold"] = threshold
        bi_json_usable["device_props"] = dict(name=platform.node(), gpu=GPU_PROPS)
        bi_json_usable["lightcone_index"] = lc_index
        bi_json_usable["bucket_index"] = bi
        bi_json_usable["opt_algo"] = opt_algo
        for i, attr in enumerate(titles):
            if attr == "flop":
                bi_json_usable["ops"] = report[i]
            else:
                bi_json_usable[attr] = report[i]
        bi_json_usable["problem"] = {
                    "n" :problem[0] , 
                    "p" :problem[1] ,
                    "d" :problem[2] ,
                    'type': problem[3]
                    }
        bi_json_usable["experiment_group"] = "Chen_Test_Seed_Fix"
        lc_collection.append(bi_json_usable)
    #print(json.dumps(lc_collection, indent = 4))

    return lc_collection














'''
testing 

'''

if __name__ == "__main__":
    # mixed_be = get_mixed_perf_backend("einsum", "cupy")
    # G, gamma, beta = get_test_problem(8,4,3,"random")
    # sim = QAOAQtreeSimulator(QtreeQAOAComposer, backend = mixed_be)
    # sim.energy_expectation(G, gamma = gamma, beta = beta)
    # mixed_be.cpu_be.gen_report(show = True)
    # mixed_be.gpu_be.gen_report(show = True)
    gen_sim = QAOAQtreeSimulator(QtreeQAOAComposer)
    my_algo = "greedy"
    backends = [["einsum", "torch_gpu",9]]
    
    for pb in [paramtest[0]]:
        with timing(callback=lambda x: None) as gen_pb:
            n, p, d, ttype = pb
            G, gamma, beta = get_test_problem(n=n, p=p,d=d,type = ttype)
            peos, widths = get_fixed_peos_for_a_pb(G, gamma, beta, algo = my_algo, sim = gen_sim)
        gen_base = gen_pb.result

        for be in [backends[0]]:
            
            for i, pack in enumerate(zip(G.edges, peos)):
                edge, peo = pack

                curr_report =process_reduced_data(G, gamma, beta, edge, peo, be, pb, 3, 114514, i, my_algo)
                for c in curr_report:
                    print(json.dumps(c))

        