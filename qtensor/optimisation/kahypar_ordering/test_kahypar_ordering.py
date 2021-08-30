#!/usr/bin/env python3
import qtensor
from qtensor import QtreeQAOAComposer
from qtensor.optimisation.Optimizer import GreedyOptimizer, TamakiOptimizer, KahyparOptimizer
from qtensor.optimisation.TensorNet import QtreeTensorNet
import numpy as np
import matplotlib.pyplot as plt 
import math
from qtensor import utils
from qtensor.tests import get_test_problem
from qtensor.Simulate import QtreeSimulator
from qtensor.optimisation.kahypar_ordering import generate_TN

# -- Timing
from contextlib import contextmanager
import time
@contextmanager
def timing():
    class Ret:
        result=None
    r = Ret()
    start = time.time()
    yield r
    end = time.time()
    r.result = end-start    
# --

np.random.seed(2021)

def test_dual_hg():
    hg = {1: [1, 2, 3], 2: [1, 4], 3: []}
    dual_expect = {1: [1, 2], 2: [1], 3: [1], 4: [2]}
    dual = generate_TN.dual_hg(hg)
    assert dual==dual_expect
        
def test_tn():
    import networkx as nx
    N = 5
    g = nx.path_graph(N)
    """
    Resulting TN (hypergraph):
        --- input ---

        |    |    |
        M    M    M
        |    |    |
        H    H    H
        |\  /|\  /|
         -ZZ- -ZZ- -
        |/  \|/  \|
        U    U    U
        |    |    |
        M    M    M
        |    |    |

        --- output ---

    """
    dangling_cnt = 2*N
    vert_cnt = 4*N + (N-1)
    edge_cnt = 5*N

    comp = qtensor.DefaultQAOAComposer(g, gamma=[1], beta=[2])
    comp.ansatz_state()
    circ = comp.circuit
    tn_ = qtensor.optimisation.QtreeTensorNet.from_qtree_gates(circ)
    print('tensors', tn_.tensors)

    tn = generate_TN.circ2tn(circ)
    #dual = generate_TN.dual_hg(tn)
    print(tn)
    dangling = [item for item in tn.values() if len(item)==1]

    assert len(dangling) == dangling_cnt
    verts = set(sum(tn.values(), []))
    assert len(verts) == vert_cnt
    assert len(tn) == edge_cnt

def test_simple_kahypar():
    ### Test without optimizer 
    #tn ={'v_1': ['A','C'], 'v_2':['A','B'], 'v_3':['B','C','D'], 
    #         'v_4':['C','E'], 'v_5':['D','F'], 'v_6':['E','F']}
    import networkx as nx
    from qtensor.optimisation.kahypar_ordering import use_kahypar
    N = 50 # the larger the harder
    p = 3 # the larger the harder
    #g = nx.path_graph(N) # simple graph structure
    g = nx.random_regular_graph(3, N) # more complicated structure
    comp = qtensor.DefaultQAOAComposer(g, gamma=[1]*p, beta=[2]*p)
    comp.ansatz_state()
    circ = comp.circuit
    tn=qtensor.optimisation.QtreeTensorNet.from_qtree_gates(circ)
    
    tn = generate_TN.circ2tn(circ)
    # preprocessing to remove edges i_ and o_ (which have only one vertex)
    edge =list(tn.keys()); edge.sort()
    rem_num_list = [*range(N), *range(len(edge)-1, len(edge)-N-1, -1)]
    rem_list = [edge[i] for i in rem_num_list]
    [tn.pop(key) for key in rem_list]
    kwargs = {'K': 5, 'eps': 0.1, 'seed': 2021, 'mode':0, 'objective':0} 
    with timing() as t_kahypar:
        tn_partite_list = use_kahypar.recur_partition(tn,**kwargs)        
        order, _ = use_kahypar.tree2order(tn,tn_partite_list) # top to bottom
    #full_order=rem_list; full_order.extend(order)
    print(order)  
    peo = [int(x) for x in order]
    print(order) 
        
def generate_problem(N,p, mode = 'ansatz'):
        ##
        G, gamma, beta = get_test_problem(N, p, d=3)
        composer = QtreeQAOAComposer(graph=G, gamma=gamma, beta=beta)
        if mode == 'ansatz':
            composer.ansatz_state()
        elif mode == 'energy':
            composer.energy_expectation_lightcone(list(G.edges())[0])
        
        ##
        #g = nx.random_regular_graph(3, N) # more complicated structure
        #comp = qtensor.DefaultQAOAComposer(g, gamma=[1]*p, beta=[2]*p)
        #comp.ansatz_state()
        
        tn = QtreeTensorNet.from_qtree_gates(composer.circuit)  
        return composer, tn

def get_tw_costs_rgreedy(tn, max_time = 1):
    ### rgreedy
    rgreedy_opt = qtensor.toolbox.get_ordering_algo('rgreedy_0.02_10', max_time=max_time)
    with timing() as t_rgreedy:
        rgreedy_peo, _ = rgreedy_opt.optimize(tn)
    rgreedy_tw = rgreedy_opt.treewidth  
    rgreedy_mems, rgreedy_flops = tn.simulation_cost(rgreedy_peo)
    return t_rgreedy.result, rgreedy_tw, rgreedy_mems, rgreedy_flops

def get_tw_costs_kahypar(tn):
    ### Kahypar
    kahypar_opt = KahyparOptimizer()   
    with timing() as t_kahypar:
        kahypar_peo, _ = kahypar_opt.optimize(tn)
    kahypar_tw = kahypar_opt.treewidth   
    kahypar_mems, kahypar_flops = tn.simulation_cost(kahypar_peo)
    return t_kahypar.result, kahypar_tw, kahypar_mems, kahypar_flops
    
def get_tw_costs_greedy(tn):
    ### Greedy
    greedy_opt = GreedyOptimizer()
    with timing() as t_greedy:
        greedy_peo, _ = greedy_opt.optimize(tn)
    greedy_tw = greedy_opt.treewidth   
    greedy_mems, greedy_flops = tn.simulation_cost(greedy_peo)
    return t_greedy.result, greedy_tw, greedy_mems, greedy_flops

def get_tw_costs_tamaki(tn, wait_time, max_tw=64):
    ### Tamaki        
    tamaki_opt = TamakiOptimizer(wait_time = wait_time) # time to run tamaki, in seconds
    tamaki_opt.max_tw = max_tw
    with timing() as t_tamaki:
        tamaki_peo, _ = tamaki_opt.optimize(tn)
    tamaki_tw = tamaki_opt.treewidth
    tamaki_time = t_tamaki.result
    tamaki_mems, tamaki_flops = tn.simulation_cost(tamaki_peo)
    return tamaki_time, tamaki_tw, tamaki_mems, tamaki_flops

def get_tw_costs_random(tn,peo):
    ### Random
    rand_tw_list = []
    with timing() as t_random:
        for _ in range(1):
            rand_peo=np.random.permutation(peo)
            nodes, ngh = utils.get_neighbors_path(tn.get_line_graph(), rand_peo)
            rand_tw_list.append(max(ngh))
    random_tw = min(rand_tw_list)
    #rand_mems, rand_flops = tn.simulation_cost(rand_peo)
    return t_random.result, random_tw
    
def test_cost_estimation_N(): 
    ### Different N
    kahypar_result, tamaki_result, greedy_result, rgreedy_result=[],[],[],[]
    mode = 'ansatz'
    N_list = [20,40,60,80,100,120]
    for N in N_list:
        p = 3 
        composer, tn = generate_problem(N,p,mode = mode)
        ##
        greedy_str = get_tw_costs_greedy(tn)
        ##
        max_time = 1
        rgreedy_str = get_tw_costs_rgreedy(tn,max_time=max_time)
        ##
        kahypar_str = get_tw_costs_kahypar(tn)
        ##
        tamaki_str=[];  wait_time_list = [30, 60, 150]
        for wait_time in wait_time_list:
            tamaki_str.append(get_tw_costs_tamaki(tn, wait_time))

        
        print(f'Problem size N={N}, p={p}, d=3')
        print('--Order Search Time--')
        print(f'Kahypa: {kahypar_str[0]}') 
        #print(f'Random: {t_random.result}') 
        for (count,wait_time) in enumerate(wait_time_list):
            print(f'Tamaki ({wait_time}): {tamaki_str[count][0]}')
        print(f'Greedy: {greedy_str[0]}')
        print(f'RGreedy: {rgreedy_str[0]}')
        print()
        
        print('--Width--')
        print(f'Partition tw: {kahypar_str[1]}') 
        #print(f'Random tw: {min(rand_tw_list)}') 
        for (count,wait_time) in enumerate(wait_time_list):
            print(f'Tamaki ({wait_time}) tw: {tamaki_str[count][1]}')
        print(f'Greedy tw: {greedy_str[1]}')
        print(f'RGreedy: {rgreedy_str[1]}')
        print()
        
        print('--Log Max memory--')
        print(f'Partition mem: {math.log2(max(kahypar_str[2]))}') 
        for (count,wait_time) in enumerate(wait_time_list):
            print(f'Tamaki ({wait_time}) mem: {math.log2(max(tamaki_str[count][2]))}')
        print(f'Greedy mem: {math.log2(max(greedy_str[2]))}')
        print(f'RGreedy mem: {math.log2(max(rgreedy_str[2]))}')
        print()
        
        print('--Log Total flops--')
        print(f'Partition flops: {math.log2(sum(kahypar_str[3]))}') 
        for (count,wait_time) in enumerate(wait_time_list):
            print(f'Tamaki ({wait_time}) flops: {math.log2(sum(tamaki_str[count][3]))}')
        print(f'Greedy flops: {math.log2(sum(greedy_str[3]))}')
        print(f'RGreedy flops: {math.log2(sum(rgreedy_str[3]))}')
        print()
        
        kahypar_result.append(kahypar_str)
        tamaki_result.append(tamaki_str)
        greedy_result.append(greedy_str)
        rgreedy_result.append(rgreedy_str)
        
    ### Plot figure
    tamaki_result = list(map(list, zip(*tamaki_result))) #important
    f, axs = plt.subplots(2, 2, figsize=(10, 10))
    f.suptitle(f'p = {p}, mode = {mode}') #fontsize=16
    plt.sca(axs[0][0])
    plt.grid()
    plt.plot(N_list,[math.log2(row[0]) for row in kahypar_result])
    for (count,_) in enumerate(wait_time_list):
        plt.plot(N_list,[math.log2(row[0]) for row in tamaki_result[count]])
    plt.plot(N_list,[math.log2(row[0]) for row in greedy_result])
    plt.plot(N_list,[math.log2(row[0]) for row in rgreedy_result])
    plt.title('Log Order Search Time')
    
    plt.sca(axs[0][1])
    plt.grid()
    plt.plot(N_list,[math.log2(row[1]) for row in kahypar_result]) 
    for (count,_) in enumerate(wait_time_list):
        plt.plot(N_list,[math.log2(row[1]) for row in tamaki_result[count]])
    plt.plot(N_list,[math.log2(row[1]) for row in greedy_result])
    plt.plot(N_list,[math.log2(row[1]) for row in rgreedy_result])
    plt.title('Log Width')
    
    plt.sca(axs[1][0])
    plt.grid()
    mem_temp=[row[2] for row in kahypar_result]
    plt.plot(N_list,[math.log2(max(x)) for x in mem_temp]) 
    for (count,_) in enumerate(wait_time_list):
        mem_temp=[row[2] for row in tamaki_result[count]]
        plt.plot(N_list,[math.log2(max(x)) for x in mem_temp])
    mem_temp=[row[2] for row in greedy_result]
    plt.plot(N_list,[math.log2(max(x)) for x in mem_temp]) 
    mem_temp=[row[2] for row in rgreedy_result]
    plt.plot(N_list,[math.log2(max(x)) for x in mem_temp]) 
    plt.title('Log Max memory')
    
    plt.sca(axs[1][1])
    plt.grid()
    flop_temp=[row[3] for row in kahypar_result]
    plt.plot(N_list,[math.log2(sum(x)) for x in flop_temp], label = 'Kahyper')
    for (count,wait_time) in enumerate(wait_time_list):
        flop_temp=[row[3] for row in tamaki_result[count]]
        plt.plot(N_list,[math.log2(max(x)) for x in flop_temp], label = 'Tamaki (%s)' % wait_time)
    flop_temp=[row[3] for row in greedy_result]
    plt.plot(N_list,[math.log2(max(x)) for x in flop_temp], label = 'Greedy')
    flop_temp=[row[3] for row in rgreedy_result]
    plt.plot(N_list,[math.log2(max(x)) for x in flop_temp], label = 'RGreedy')
    plt.title('Log Total flops')  
    plt.legend()
    
    plt.show() # shows a plot if run from terminal on machine with GUI on
    
def test_cost_estimation_p(): 
    ### Different p
    kahypar_result, tamaki_result, greedy_result, rgreedy_result=[],[],[],[]
    mode = 'ansatz'
    p_list = [2,4,6,8,10]
    for p in p_list:
        N = 60
        composer, tn = generate_problem(N,p,mode = mode)
        kahypar_str = get_tw_costs_kahypar(tn)
        tamaki_str=[];  wait_time_list = [30,60,150]
        for wait_time in wait_time_list:
            tamaki_str.append(get_tw_costs_tamaki(tn, wait_time))
        greedy_str = get_tw_costs_greedy(tn)
        max_time = 1
        rgreedy_str = get_tw_costs_rgreedy(tn,max_time=max_time)
        
        print(f'Problem size N={N}, p={p}, d=3')
        print('--Order Search Time--')
        print(f'Kahypa: {kahypar_str[0]}') 
        #print(f'Random: {t_random.result}') 
        for (count,wait_time) in enumerate(wait_time_list):
            print(f'Tamaki ({wait_time}): {tamaki_str[count][0]}')
        print(f'Greedy: {greedy_str[0]}')
        print(f'RGreedy: {rgreedy_str[0]}')
        print()
        
        print('--Width--')
        print(f'Partition tw: {kahypar_str[1]}') 
        #print(f'Random tw: {min(rand_tw_list)}') 
        for (count,wait_time) in enumerate(wait_time_list):
            print(f'Tamaki ({wait_time}) tw: {tamaki_str[count][1]}')
        print(f'Greedy tw: {greedy_str[1]}')
        print(f'RGreedy: {rgreedy_str[1]}')
        print()
        
        print('--Log Max memory--')
        print(f'Partition mem: {math.log2(max(kahypar_str[2]))}') 
        for (count,wait_time) in enumerate(wait_time_list):
            print(f'Tamaki ({wait_time}) mem: {math.log2(max(tamaki_str[count][2]))}')
        print(f'Greedy mem: {math.log2(max(greedy_str[2]))}')
        print(f'RGreedy mem: {math.log2(max(rgreedy_str[2]))}')
        print()
        
        print('--Log Total flops--')
        print(f'Partition flops: {math.log2(sum(kahypar_str[3]))}') 
        for (count,wait_time) in enumerate(wait_time_list):
            print(f'Tamaki ({wait_time}) flops: {math.log2(sum(tamaki_str[count][3]))}')
        print(f'Greedy flops: {math.log2(sum(greedy_str[3]))}')
        print(f'RGreedy flops: {math.log2(sum(rgreedy_str[3]))}')
        print()
        
        kahypar_result.append(kahypar_str)
        tamaki_result.append(tamaki_str)
        greedy_result.append(greedy_str)
        rgreedy_result.append(rgreedy_str)
    ### Plot figure
    tamaki_result = list(map(list, zip(*tamaki_result))) #important
    f, axs = plt.subplots(2, 2, figsize=(10, 10))
    f.suptitle(f'N = {N}, mode = {mode}') #fontsize=16
    plt.sca(axs[0][0])
    plt.grid()
    plt.plot(p_list,[math.log2(row[0]) for row in kahypar_result])
    for (count,_) in enumerate(wait_time_list):
        plt.plot(p_list,[math.log2(row[0]) for row in tamaki_result[count]])
    plt.plot(p_list,[math.log2(row[0]) for row in greedy_result])
    plt.plot(p_list,[math.log2(row[0]) for row in rgreedy_result])
    plt.title('Log Order Search Time')
    
    plt.sca(axs[0][1])
    plt.grid()
    plt.plot(p_list,[math.log2(row[1]) for row in kahypar_result]) 
    for (count,_) in enumerate(wait_time_list):
        plt.plot(p_list,[math.log2(row[1]) for row in tamaki_result[count]])
    plt.plot(p_list,[math.log2(row[1]) for row in greedy_result])
    plt.plot(p_list,[math.log2(row[1]) for row in rgreedy_result])
    plt.title('Log Width')
    
    plt.sca(axs[1][0])
    plt.grid()
    mem_temp=[row[2] for row in kahypar_result]
    plt.plot(p_list,[math.log2(max(x)) for x in mem_temp]) 
    for (count,_) in enumerate(wait_time_list):
        mem_temp=[row[2] for row in tamaki_result[count]]
        plt.plot(p_list,[math.log2(max(x)) for x in mem_temp])
    mem_temp=[row[2] for row in greedy_result]
    plt.plot(p_list,[math.log2(max(x)) for x in mem_temp])
    mem_temp=[row[2] for row in rgreedy_result]
    plt.plot(p_list,[math.log2(max(x)) for x in mem_temp]) 
    plt.title('Log Max memory')
    
    plt.sca(axs[1][1])
    plt.grid()
    flop_temp=[row[3] for row in kahypar_result]
    plt.plot(p_list,[math.log2(sum(x)) for x in flop_temp], label = 'Kahyper')
    for (count,wait_time) in enumerate(wait_time_list):
        flop_temp=[row[3] for row in tamaki_result[count]]
        plt.plot(p_list,[math.log2(max(x)) for x in flop_temp], label = 'Tamaki (%s)' % wait_time)
    flop_temp=[row[3] for row in greedy_result]
    plt.plot(p_list,[math.log2(max(x)) for x in flop_temp], label = 'Greedy')
    flop_temp=[row[3] for row in rgreedy_result]
    plt.plot(p_list,[math.log2(max(x)) for x in flop_temp], label = 'RGreedy')
    plt.title('Log Total flops')
    plt.legend()
    
    plt.show() # shows a plot if run from terminal on machine with GUI on
                 
def test_get_tw():
    N_list = list(range(10, 100+10, 10))
    N_list.extend(list(range(200, 1000+100, 100)))
    p_list = [1,2,3,4,5]
    mode = 'ansatz'
    kahypar_tw_list = []
    for (count,p) in enumerate(p_list):
        kahypar_tw_list.append([])
        for N in N_list:
            composer, tn = generate_problem(N,p,mode = mode)
            kahypar_str = get_tw_costs_kahypar(tn)   
            kahypar_tw_list[count].append(kahypar_str[1])
            
    ### plot
    plt.figure()
    for (count,p) in enumerate(p_list):
        plt.plot(N_list, kahypar_tw_list[count], label = 'p = %s' % p)
    plt.ylabel('Contraction width')
    plt.xscale('log')
    plt.xlabel('N')
    plt.title(f'mode = {mode}')
    plt.legend()
    plt.grid()
    plt.show()
    
def test_qtree():
    N = 300 
    p_list = [1,2,3,4]
    mode = 'energy'
    wait_time = 1
    t_tamaki_list = []; t_greedy_list = []; t_kahypar_list = []
    for (count,p) in enumerate(p_list):
        composer, tn = generate_problem(N,p,mode=mode)
        ###
        optimizer=TamakiOptimizer(wait_time = wait_time)
        sim = QtreeSimulator(optimizer = optimizer)
        with timing() as t_tamaki:
            result_tamaki = sim.simulate(composer.circuit)
        
        optimizer=GreedyOptimizer()
        sim = QtreeSimulator(optimizer = optimizer)
        with timing() as t_greedy:
            result_greedy = sim.simulate(composer.circuit)
        
        assert np.allclose(result_tamaki, result_greedy)
        
        optimizer=KahyparOptimizer()
        sim = QtreeSimulator(optimizer = optimizer)
        with timing() as t_kahypar:
            result_kahypar = sim.simulate(composer.circuit)
            
        assert np.allclose(result_greedy, result_kahypar)
        
        print('--Simulation Time--')
        print(f'Tamaki: {t_tamaki.result}')  
        print(f'Greedy: {t_greedy.result}')
        print(f'Kahypar: {t_kahypar.result}')  
        t_tamaki_list.append(t_tamaki.result)
        t_greedy_list.append(t_greedy.result)
        t_kahypar_list.append(t_kahypar.result)
        
    plt.figure()
    plt.plot(p_list, t_tamaki_list, label = 'Tamaki (%s)' % wait_time)
    plt.plot(p_list, t_greedy_list, label = 'Greedy')
    plt.plot(p_list, t_kahypar_list, label = 'Kahypar')
    plt.ylabel('Simulation time')
    plt.yscale('log')
    plt.xlabel('p')
    plt.title(f'N = {N}, mode = {mode}')
    plt.xticks(p_list,p_list)
    plt.legend()
    plt.grid()
    plt.show()
   
            
            
if __name__ == '__main__':
    #test_simple_kahypar()
    test_cost_estimation_N()
    #test_cost_estimation_p()
    #test_get_tw()
    #test_qtree()