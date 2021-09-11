#!/usr/bin/env python3
import qtensor
from qtensor.optimisation.Optimizer import GreedyOptimizer, TamakiOptimizer, KahyparOptimizer
import numpy as np
import json
import platform
from qtensor.Simulate import QtreeSimulator
from qtensor.optimisation.kahypar_ordering.test_kahypar_ordering import get_tw_costs_kahypar, \
get_tw_costs_greedy, get_tw_costs_rgreedy, get_tw_costs_tamaki, generate_problem, \
    timing
np.random.seed(2021)

### json format
def print_results_json(mode, N, p, method, result, func):
    res = dict(
                mode=mode
                ,device_props=dict(name=platform.node())
                ,N=N
                ,p=p
                ,method=method
                ,time=result[0]
                ,tw=int(result[1])
                ,mem=max(result[2])
                ,flop=sum(result[3])
                ,func=func)
    #print(json.dumps(res), flush=True)
    return res


def test_cost_estimation(): ### Different mode, p, N
    mode_list = ['ansatz'] #energy
    p_list = [3] #[2,4,6,8,10]
    N_list = [20,40,60,80,100,120]
    func_name = test_cost_estimation.__name__
    with open('test_cost_estimation_p3.jsonl', 'w') as f:
        for mode in mode_list:
            for p in p_list:
                for N in N_list:
                    composer, tn = generate_problem(N,p,mode = mode)
                    ###
                    rgreedy_str = get_tw_costs_rgreedy(tn)
                    #print_results_json(mode, N, p, 'RGreedy', greedy_str)
                    json.dump(print_results_json(mode, N, p, 'RGreedy', rgreedy_str,func_name),f)
                    f.write('\n')
                    
                    ###
                    greedy_str = get_tw_costs_greedy(tn)
                    #print_results_json(mode, N, p, 'Greedy', greedy_str)
                    json.dump(print_results_json(mode, N, p, 'Greedy', greedy_str,func_name),f)
                    f.write('\n')
                    
                    ###
                    kahypar_str = get_tw_costs_kahypar(tn)
                    #print_results_json(mode, N, p, 'Kahypar', kahypar_str)
                    json.dump(print_results_json(mode, N, p, 'Kahypar', kahypar_str,func_name),f)
                    f.write('\n')
                    
                    ###
                    tamaki_str=[];  wait_time_list = [30,60,150]
                    for (count,wait_time) in enumerate(wait_time_list):
                        tamaki_str=get_tw_costs_tamaki(tn, wait_time)
                        name = 'Tamaki ({:d})'.format(wait_time)
                        #print_results_json(mode, N, p, name, tamaki_str[count])
                        json.dump(print_results_json(mode, N, p, name, tamaki_str,func_name),f)
                        f.write('\n')
            
 
                 
def test_get_tw():
    mode_list = ['ansatz','energy']
    N_list = list(range(10, 100+10, 10))
    N_list.extend(list(range(200, 1000+100, 100)))
    p_list = [1,2,3,4,5]
    func_name = test_get_tw.__name__
    with open('test_get_tw.jsonl', 'w') as f: 
        for mode in mode_list:
            for p in p_list:
                for N in N_list:
                    composer, tn = generate_problem(N,p,mode = mode)
                    ###
                    kahypar_str = get_tw_costs_kahypar(tn)   
                    #print_results_json(mode, N, p, 'Kahypar', kahypar_str)
                    json.dump(print_results_json(mode, N, p, 'Kahypar', kahypar_str,func_name),f)
                    f.write('\n')
                    
                    ###
                    rgreedy_str = get_tw_costs_rgreedy(tn)
                    json.dump(print_results_json(mode, N, p, 'RGreedy', rgreedy_str,func_name),f)
                    f.write('\n')
    
    
    
def test_qtree():
    mode_list = ['energy']
    N_list = list(range(10, 100+10, 10))
    N_list.extend(list(range(200, 1000+100, 100)))
    #N_list = [50,100,200,300] 
    p_list = [1,2,3,4]
    func_name = test_qtree.__name__
    def qtree_results_json(mode, N, p, method, result, func):
        res = dict(
                    mode=mode
                    ,device_props=dict(name=platform.node())
                    ,N=N
                    ,p=p
                    ,method=method
                    ,time=result
                    ,func=func)
        #print(json.dumps(res), flush=True)
        return res

    with open('test_qtree.jsonl', 'w') as f: 
        for mode in mode_list:
            for p in p_list:
                for N in N_list:
                    composer, tn = generate_problem(N,p,mode=mode)
                    ###
                    wait_time = 1
                    optimizer=TamakiOptimizer(wait_time = wait_time)
                    sim = QtreeSimulator(optimizer = optimizer)
                    with timing() as t_tamaki:
                        result_tamaki = sim.simulate(composer.circuit)
                    name = 'Tamaki ({:d})'.format(wait_time)
                    json.dump(qtree_results_json(mode, N, p, name, t_tamaki.result,func_name),f)
                    f.write('\n')
                    
                    optimizer=GreedyOptimizer()
                    sim = QtreeSimulator(optimizer = optimizer)
                    with timing() as t_greedy:
                        result_greedy = sim.simulate(composer.circuit)
                    json.dump(qtree_results_json(mode, N, p, 'Greedy', t_greedy.result,func_name),f)
                    f.write('\n')
                    
                    assert np.allclose(result_tamaki, result_greedy)
                    
                    max_time = 1
                    optimizer=qtensor.toolbox.get_ordering_algo('rgreedy_0.02_10', max_time=max_time)
                    sim = QtreeSimulator(optimizer = optimizer)
                    with timing() as t_rgreedy:
                        result_rgreedy = sim.simulate(composer.circuit)
                    json.dump(qtree_results_json(mode, N, p, 'RGreedy', t_rgreedy.result, func_name),f)
                    f.write('\n')
                    
                    assert np.allclose(result_rgreedy, result_greedy)
                    
                    optimizer=KahyparOptimizer()
                    sim = QtreeSimulator(optimizer = optimizer)
                    with timing() as t_kahypar:
                        result_kahypar = sim.simulate(composer.circuit)
                    json.dump(qtree_results_json(mode, N, p, 'Kahypar', t_kahypar.result,func_name),f)
                    f.write('\n')
                    
                    assert np.allclose(result_greedy, result_kahypar)
        
def test_get_tw_energy_best():
    ##
    from qtensor.tests import get_test_problem
    
    def energy_best_results_json(mode, N, p, method, result, lightcone_ind, func):
        res = dict(
                    mode=mode
                    ,device_props=dict(name=platform.node())
                    ,N=N
                    ,p=p
                    ,method=method
                    ,time=result[0]
                    ,tw=int(result[1])
                    ,lightcone_ind=lightcone_ind
                    ,func=func)
        print(json.dumps(res), flush=True)
        #return res
    
    N_list = list(range(10, 100+10, 10))
    N_list.extend(list(range(200, 1000+100, 100)))
    p_list=[1,2,3,4,5] 
    mode = 'energy_best'
    func_name = test_get_tw_energy_best.__name__
    for (count,p) in enumerate(p_list):
        for N in N_list:
            G, gamma, beta = get_test_problem(N, p, d=3)
            composer = qtensor.DefaultQAOAComposer(graph=G, gamma=gamma, beta=beta)
            #greedy_list, rgreedy_list, kahypar_list = [], [], []
            for (lightcone_ind,edge) in enumerate(G.edges()):
                composer.energy_expectation_lightcone(edge)
                #tn = QtreeTensorNet.from_qtree_gates(composer.circuit)  
                tn = qtensor.optimisation.QtreeTensorNet.from_qtree_gates(composer.circuit)
                ##
                #greedy_list.append(get_tw_costs_greedy(tn))
                greedy_str = get_tw_costs_greedy(tn)
                energy_best_results_json(mode, N, p, 'Greedy', greedy_str,lightcone_ind,func_name)
                ##
                max_time = 1
                #rgreedy_list.append(get_tw_costs_rgreedy(tn,max_time=max_time))
                rgreedy_str = get_tw_costs_rgreedy(tn,max_time=max_time)
                energy_best_results_json(mode, N, p, 'RGreedy', rgreedy_str,lightcone_ind,func_name)
                ##
                #kahypar_list.append(get_tw_costs_kahypar(tn))
                kahypar_str = get_tw_costs_kahypar(tn)
                energy_best_results_json(mode, N, p, 'Kahypar', kahypar_str,lightcone_ind,func_name)
                
    # ### plot
    # plt.figure()
    # ax = plt.gca()
    # for (count,p) in enumerate(p_list):
    #     color = next(ax._get_lines.prop_cycler)['color']
    #     plt.plot(N_list, kahypar_result[count], linestyle='-', color = color, label = 'Kahypar p = %s' % p)
    #     plt.plot(N_list, rgreedy_result[count], linestyle='--', color = color, label = 'RGreedy p = %s' % p)
    # plt.ylabel('TW')
    # plt.xscale('log')
    # plt.xlabel('N')
    # plt.title(f'mode = {mode}')
    # plt.legend()
    # plt.grid()
    # plt.show()  
    
if __name__ == '__main__':
    #test_cost_estimation()
    test_get_tw_energy_best()
    #test_get_tw()
    #test_qtree()
