import json
from functools import lru_cache
import numpy as np
import networkx as nx
import platform
import pyrofiler
import qtensor
from qtensor import QtreeQAOAComposer
from qtensor import QAOAQtreeSimulator
from qtensor.contraction_backends import get_cpu_perf_backend, get_gpu_perf_backend
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sklearn.metrics import r2_score
from scipy.stats import chisquare, pearsonr

gpu_backends = ['torch_gpu', 'cupy', 'tr_torch', 'tr_cupy', 'tr_cutensor']
cpu_backends = ['einsum', 'torch_cpu', 'mkl', 'opt_einsum', 'tr_einsum', 'opt_einsum']
timing = pyrofiler.timing

def func(x,a,b,c,d):
    # return (a*np.power(x1,-n))
    return a*np.power(b, c*x) + d

def diff(x, OPT1, OPT2):
    return func(x, *OPT1) - func(x, *OPT2)

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
    [12,4,3,"random"]
    ,[10, 5, 2, 'random']
    ,[14, 1, 3, 'random']
    # ,[3, 3, 0, 'grid2d']
    # ,[8, 4, 0, 'line']
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

    if backend_name in gpu_backends:
        curr_backend = get_gpu_perf_backend(backend_name)
    else:
        curr_backend = get_cpu_perf_backend(backend_name)
    curr_sim = QAOAQtreeSimulator(QtreeQAOAComposer,backend=curr_backend)
    circuit = curr_sim._edge_energy_circuit(G, gamma, beta, edge)
    curr_sim.simulate_batch(circuit, peo = peo)
    curr_sim.backend.gen_report(show = False)

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
    
    report_record = curr_sim.backend.report_table.records
    gen_time = gen_base
    title_record = curr_sim.backend.report_table._title_row()[1:]

    '''
    Modify and concatenate new report table
    '''
    title_record.append("byte")
    title_record.append("gen_time")
    for i, row in enumerate(report_record):
        row.append(tensor_dims[i])
        row.append(gen_time)

    return title_record, report_record


'''
Function: bucket_index -> [repeat] amount of reports for this particular bucket
'''
def collect_reports_for_a_lc(G, gamma, beta, edge, peo, backend_name, repeat, gen_base):
    bucketindex_2_reports = {}
    '''
    Collect each lc's report for [repeat] times
    '''
    for _ in range(repeat):
        titles, report_table = gen_be_lc_report(G, gamma, beta, edge, peo, backend_name, gen_base)
        for i, report in enumerate(report_table):
            if i not in bucketindex_2_reports:
                bucketindex_2_reports[i] = [report]
            else:
                bucketindex_2_reports[i].append(report)
    
    #print(bucketindex_2_reports)
    return titles, bucketindex_2_reports

'''
Function: run above functions, collect reports, and reduce the data
'''
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

'''
Function: Takes in reduced data form, and process them into json format, with knowledge regarding lightcone index and bucket index
'''
def process_reduced_data(G, gamma, beta, edge, peo, backend_name, problem, repeat, gen_base, lc_index, opt_algo):
    titles, bi_2_reduced = reduce_bucket_reports(G, gamma, beta, edge, peo, backend_name, repeat, gen_base)
    GPU_PROPS = get_gpu_props_json()
    lc_collection = []
    for bi, report in bi_2_reduced.items():
        bi_json_usable = {}
        bi_json_usable["backend"] = backend_name
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
        bi_json_usable["experiment_group"] = "Angela_nslb_circuit"
        lc_collection.append(bi_json_usable)
    #print(json.dumps(lc_collection, indent = 4))

    return lc_collection
    

'''
Function: Takes in the total report for a backend, and generate the array_index -> sum time
Update1: Change sum time to mean time
'''
def process_all_lc_reports(be_lc_reports:list):

    width2times = {}

    for lc in be_lc_reports:
        for bucket in lc:
            width = bucket["width"]
            time = bucket["time"]
            if width not in width2times:
                width2times[width] = [time]
            else:
                width2times[width].append(time)
    
    width2meanTime = [0] * len(width2times)
    for width, times in width2times.items():
        width2meanTime[int(width-1)] = sum(times)/len(times)

    return width2meanTime

'''
Fundtion: Takes in a dict of be->time distro, find the best threshold between cpu and gpu, single log transforming y
'''
def threshold_finding(dict_of_distro:dict):
    
    '''
    1. Obtain Original Array
    '''
    npDistro = dict_of_distro["einsum"]
    cpDistro = dict_of_distro["cupy"]
    tcpuDistro = dict_of_distro["torch_cpu"]
    tgpuDistro = dict_of_distro["torch_gpu"]


    '''
    2. Transform into natural log scale
    '''
    npLog = np.log(npDistro)
    cpLog = np.log(cpDistro)
    tcpuLog =np.log(tcpuDistro)
    tgpuLog = np.log(tgpuDistro)

    '''
    3. Apply Shifting to avoid negative value
    '''
    shift = min(min(npLog), min(cpLog), min(tcpuLog), min(tgpuLog))

    npFinal = npLog - shift
    cpFinal = cpLog - shift
    tcpuFinal = tcpuLog - shift
    tgpuFinal = tgpuLog - shift

    '''
    4. Curve Fitting to original X
    '''
    X = np.arange(1,len(npDistro)+1)
    npOPT, npCOV = curve_fit(func, X, npFinal, maxfev=1000000)
    cpOPT, cpCOV = curve_fit(func, X, cpFinal, maxfev=1000000)
    tcpuOPT, tcpuCOV = curve_fit(func, X, tcpuFinal, maxfev=1000000)
    tgpuOPT, tgpuCOV = curve_fit(func, X, tgpuFinal, maxfev=1000000)
    

    '''
    3. Plotting X vs. Final Log Transformation
    '''
    plt.scatter(X, npFinal, color = 'blue', s = 2)
    plt.scatter(X, cpFinal, color = 'green', s=2)
    plt.scatter(X, tcpuFinal, color = 'black', s=2)
    plt.scatter(X, tgpuFinal, color = 'red',s=2)

    '''
    4. Plotting Line
    '''
    plt.plot(X, func(X, *npOPT), color="blue", linewidth=0.5, label = "Numpy: {:0.3e}*{:0.3f}^({:0.3f}*x) + {:0.3f}".format(*npOPT))
    plt.plot(X, func(X, *cpOPT), color="green", linewidth=0.5, label = "CuPy: {:0.3e}*{:0.3f}^({:0.3f}*x) + {:0.3f}".format(*cpOPT))
    plt.plot(X, func(X, *tcpuOPT), color="black", linewidth=0.5, label = "TCPU: {:0.3e}*{:0.3f}^({:0.3f}*x) + {:0.3f}".format(*tcpuOPT))
    plt.plot(X, func(X, *tgpuOPT), color="red", linewidth=0.5, label = "TGPU: {:0.3e}*{:0.3f}^({:0.3f}*x) + {:0.3f}".format(*tgpuOPT))

    plt.xticks(X)
    plt.yticks()
    plt.xlabel('Width')
    plt.legend(loc = "upper left", prop={'size': 6})
    plt.title("Mean Time vs Width")
    plt.savefig("Probing.png")

    print("Numpy: {:0.3e}*{:0.3f}^({:0.3f}*x) + {:0.3f}".format(*npOPT))
    print("CuPy: {:0.3e}*{:0.3f}^({:0.3f}*x) + {:0.3f}".format(*cpOPT))
    print("TCPU: {:0.3e}*{:0.3f}^({:0.3f}*x) + {:0.3f}".format(*tcpuOPT))
    print("TGPU: {:0.3e}*{:0.3f}^({:0.3f}*x) + {:0.3f}".format(*tgpuOPT))

    '''
    5. Output threshold information
    TODO: SOLVE THE EQUATIO
    '''
    np_cpT = fsolve(diff,10, args=(npOPT, cpOPT))
    np_tgpuT = fsolve(diff,10, args=(npOPT, tgpuOPT))
    tcpu_cpT = fsolve(diff,10, args=(tcpuOPT, cpOPT))
    tcpu_tgpuT = fsolve(diff,10, args=(tcpuOPT, tgpuOPT))

    print("Numpy-Cupy Threshold is {}.".format(np_cpT))
    print("Numpy-TGPU Threshold is {}.".format(np_tgpuT))
    print("TCPU-Cupy Threshold is {}.".format(tcpu_cpT))
    print("TCPU-TGPU Threshold is {}.".format(tcpu_tgpuT))

    '''
    6. Output Strength Information
    '''
    npPred = np.array([func(i, *npOPT) for i in X])
    cpPred = np.array([func(i, *cpOPT) for i in X])
    tcpuPred = np.array([func(i, *tcpuOPT) for i in X])
    tgpuPred = np.array([func(i, *tgpuOPT) for i in X])

    npR2 = r2_score(npFinal, npPred)
    cpR2 = r2_score(cpFinal, cpPred)
    tcpuR2 = r2_score(tcpuFinal, tcpuPred)
    tgpuR2 = r2_score(tgpuFinal, tgpuPred)

    npChi2, npChiP = chisquare(npFinal, npPred)
    cpChi2, cpChiP = chisquare(cpFinal, cpPred)
    tcpuChi2, tcpuChiP = chisquare(tcpuFinal, tcpuPred)
    tgpuChi2, tgpuChiP = chisquare(tcpuFinal, tgpuPred)

    npPear, npPear_p = pearsonr(X, npPred)
    cpPear, cpPear_p = pearsonr(X, cpPred)
    tcpuPear, tcpuPear_p = pearsonr(X, tcpuPred)
    tgpuPear, tgpuPear_p = pearsonr(X, tgpuPred)

    print("Numpy Test: R2={:0.3f} Chi2={:0.3f} Pearson={:0.3f}".format(npR2, npChi2, npPear))
    print("CuPy Test: R2={:0.3f} Chi2={:0.3f} Pearson={:0.3f}".format(cpR2, cpChi2, cpPear))
    print("TCPU Test: R2={:0.3f} Chi2={:0.3f} Pearson={:0.3f}".format(tcpuR2, tcpuChi2, tcpuPear))
    print("TGPU Test: R2={:0.3f} Chi2={:0.3f} Pearson={:0.3f}".format(tgpuR2, tgpuChi2, tgpuPear))
    


if __name__ == '__main__':
    gen_sim = QAOAQtreeSimulator(QtreeQAOAComposer)
    backends = ['einsum', 'torch_cpu', 'torch_gpu', 'cupy'] #'tr_torch'

    my_algo = 'greedy'

    for pb in [paramtest[0]]:

        '''
        Generate fixed peos for a given problem, thus be used for various backends
        '''
        with timing(callback=lambda x: None) as gen_pb:
            n, p, d, ttype = pb
            G, gamma, beta = get_test_problem(n=n,p=p,d=d, type = ttype)
            peos, widths = get_fixed_peos_for_a_pb(G, gamma, beta, algo = my_algo, sim = gen_sim)

        gen_base = gen_pb.result
        be2timeDistro = {}
        for be in backends:
            all_lightcones_report = []
            for i, pack in enumerate(zip(G.edges, peos)):
                edge, peo = pack
                curr_report = process_reduced_data(G, gamma, beta, edge, peo, be, pb, 5, gen_base, i, my_algo)
                all_lightcones_report.append(curr_report)
            distro = process_all_lc_reports(all_lightcones_report)
            be2timeDistro[be] = distro
        threshold_finding(be2timeDistro)