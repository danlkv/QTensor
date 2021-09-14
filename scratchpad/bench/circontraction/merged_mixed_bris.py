import json
from functools import lru_cache
import numpy as np
import networkx as nx
import platform
import pyrofiler
import qtensor
from qtensor import QtreeQAOAComposer
from qtensor import QAOAQtreeSimulator
from qtensor.contraction_backends import get_mixed_backend, get_mixed_perf_backend, get_gpu_perf_backend, get_cpu_perf_backend
from qtensor.tools.benchmarking import qc

from qtensor.MergedSimulator import MergedQAOASimulator

'''
Helper Functions

'''
gpu_backends = ['torch_gpu', 'cupy', 'tr_torch', 'tr_cupy', 'tr_cutensor']
cpu_backends = ['einsum', 'torch_cpu', 'mkl', 'opt_einsum', 'tr_einsum', 'opt_einsum']
timing = pyrofiler.timing

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
    [6,40,0]
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
TODO:
Objective 1:
I/O: problem description -> circ collection
'''
def gen_bris_circ(d,l,s):
    _, circ = qc.get_bris_circuit(d,l,s)
    circ = sum(circ, [])
    return circ

'''
TODO:
I/O: a circuit -> gets peos
'''
def gen_circ_peo(circuit, algo):
    tn = qtensor.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(circuit)
    opt = qtensor.toolbox.get_ordering_algo(algo)
    peo, _ = opt.optimize(tn)
    return peo

'''
TODO:
I/O: Running the actual simulaton for pure backend
'''
def gen_circ_peo_report(circuit, peo, backend_name, gen_base, merged):
    with timing(callback = lambda x: None) as gen:
        if backend_name in gpu_backends:
            curr_backend = get_gpu_perf_backend(backend_name)
        else:
            curr_backend = get_cpu_perf_backend(backend_name)
    if merged:
        curr_sim = MergedQAOASimulator(QtreeQAOAComposer,backend=curr_backend)
    else:
        curr_sim = QAOAQtreeSimulator(QtreeQAOAComposer,backend=curr_backend)
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


def gen_circ_peo_report_mix(circuit, peo, backend_name, gen_base, merged):

    curr_backend = get_mixed_perf_backend(backend_name[0], backend_name[1])
    if merged:
        curr_sim = MergedQAOASimulator(QtreeQAOAComposer, backend = curr_backend)
    else:
        curr_sim = QAOAQtreeSimulator(QtreeQAOAComposer,backend=curr_backend)
    
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

def collect_reports(circuit, peo, backend_name, repeat, gen_base):
    bucketindex_2_reports = {}
    for _ in range(repeat):
        if type(backend_name) == list:
            if backend_name[-1] == "merged":
                if len(backend_name) == 2:
                    titles, report_table = gen_circ_peo_report(circuit, peo, backend_name, gen_base, True)
                else:
                    titles, report_table = gen_circ_peo_report_mix(circuit, peo, backend_name, gen_base, True)
            else:
                titles, report_table = gen_circ_peo_report_mix(circuit, peo, backend_name, gen_base, False)
        else:
            titles, report_table = gen_circ_peo_report(circuit, peo, backend_name, gen_base, False)
        
        for i, report in enumerate(report_table):
            if i not in bucketindex_2_reports:
                bucketindex_2_reports[i] = [report]
            else:
                bucketindex_2_reports[i].append(report)
    
    return titles, bucketindex_2_reports

def reduce_reports(circuit, peo, backend_name, repeat, gen_base):
    bi_2_reduced = {}
    titles, bi_2_reports = collect_reports(circuit, peo, backend_name, repeat, gen_base)

    for bi, report in bi_2_reports.items():
        bi_redux = list()
        t_report = np.transpose(report)
        for attr in t_report:
            bi_redux.append(mean_mmax(attr.tolist()))
        bi_2_reduced[bi] = bi_redux
    
    return titles, bi_2_reduced

def process_reduced_data(circuit, peo, backend_name, problem, repeat, gen_base,  opt_algo):
    if type(backend_name) == list:
        if backend_name[-1] == "merged":
            if len(backend_name) == 2:
                final_backend_name = backend_name[0] +"-merged"
            else:
                final_backend_name = backend_name[0]+"-"+backend_name[1] + "-merged"
        else:
            final_backend_name = backend_name[0]+"-"+backend_name[1]
    else:
        final_backend_name = backend_name
    titles, bi_2_reduced = reduce_reports(circuit, peo, backend_name, repeat, gen_base)
    GPU_PROPS = get_gpu_props_json()
    lc_collection = []
    for bi, report in bi_2_reduced.items():
        bi_json_usable = {}
        bi_json_usable["backend"] = final_backend_name
        bi_json_usable["device_props"] = dict(name=platform.node(), gpu=GPU_PROPS)
        bi_json_usable["bucket_index"] = bi
        bi_json_usable["opt_algo"] = opt_algo
        for i, attr in enumerate(titles):
            if attr == "flop":
                bi_json_usable["ops"] = report[i]
            else:
                bi_json_usable[attr] = report[i]
        bi_json_usable["problem"] = {
                    "d" :problem[0] , 
                    "l" :problem[1] ,
                    "s" :problem[2]
                    }
        bi_json_usable["experiment_group"] = "Chen_Bris_Test"
        lc_collection.append(bi_json_usable)
    #print(json.dumps(lc_collection, indent = 4))

    return lc_collection

    




if __name__ == '__main__':
    backends = [["torch_cpu","torch_gpu",12,"merged"]] #'tr_torch'
    my_algo = 'greedy'
    my_reap = 3
    for pb in [paramtest[0]]:
        d,l,s = pb
        with timing(callback=lambda x: None) as gen_pb:
            curr_circ = gen_bris_circ(d,l,s)
            curr_peo = gen_circ_peo(curr_circ, my_algo)
        gen_base = gen_pb.result
        for be in [backends[0]]:
            lc_collection = process_reduced_data(curr_circ, curr_peo, be, pb, my_reap, gen_base, my_algo)
            for c in lc_collection:
                print(json.dumps(c))