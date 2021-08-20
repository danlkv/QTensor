<<<<<<< HEAD
"""
This module implements interface to KaHypar program.
"""
=======
>>>>>>> 10bcdad5666d7369b4cd764d756886391a73650e
import qtensor
from qtensor.optimisation.kahypar_ordering import generate_TN
import kahypar as kahypar
from os.path import join, abspath, dirname

def set_context(**kwargs):
        mode = modes[int(kwargs.get('mode'))]
        objective = objectives[int(kwargs.get('objective'))]
        K = int(kwargs.get('K'))
        eps = kwargs.get('eps')
<<<<<<< HEAD
        seed = kwargs.get('seed')
=======
        #seed = kwargs.get('seed')
>>>>>>> 10bcdad5666d7369b4cd764d756886391a73650e
        profile_mode = {'direct': 'k', 'recursive': 'r'}[mode]
        profile = f"{objective}_{profile_mode}KaHyPar_sea20.ini"
        context = kahypar.Context()
        context.loadINIconfiguration(join(KAHYPAR_PROFILE_DIR, profile))
        context.setK(K)
<<<<<<< HEAD
        context.setSeed(seed)
        context.setEpsilon(kwargs.get('epsilon', eps * (K - 1))) #imbalance
        context.suppressOutput(True)
=======
        #context.setSeed(seed)
        context.setEpsilon(kwargs.get('epsilon', eps * (K - 1)))
        #context.suppressOutput(kwargs.get('quiet', True))
>>>>>>> 10bcdad5666d7369b4cd764d756886391a73650e
        return context

def ka_hg_init(tn): # a dictionary from circ2tn
    h = tn.values()
    l = list({l for word in h for l in word}) #set of unique edges (eg, v_1)
    l.sort()
    nodes = list(range(0, len(l)))
    edges = []
    hyperedge_indices = [0]
    hyperedges = []
    edge_weights = []
    node_weights = []
    for count, value in enumerate(h):
        edges.append([l.index(i) for i in value])
        hyperedges += edges[count]
        hyperedge_indices.append(len(hyperedges))
        edge_weights.append(1)
        
    return nodes, edges, hyperedge_indices, hyperedges, edge_weights, node_weights, l
     
def num_same_element(a:list, b:list):
    num = 0
    for i in a:
<<<<<<< HEAD
            num = num + 1
    return num    

def single_partition(tn,**kwargs): 
    K = int(kwargs.get('K'))
    nodes, edges, hyperedge_indices, hyperedges, edge_weights, node_weights, l = ka_hg_init(tn)
    hypergraph = kahypar.Hypergraph(len(nodes), len(edges), hyperedge_indices, hyperedges, K, edge_weights, node_weights)
    
    ### Set context
    # context = set_context(K = K, eps = 0.03, mode = 1, objective = 0)    
    context = set_context(**kwargs)
    
    ### perform partition
    kahypar.partition(hypergraph, context)
    partitions_names = [[] for _ in range(K)]
    for i, n in list(enumerate(nodes)):
        partitions_names[hypergraph.blockID(i)].append(n)
        
    return partitions_names, l

def subgraph_partition(tn,**kwargs):
    tn_partite_list = []
    # construct subgraphs
    partitions_names, vertex_list = single_partition(tn,**kwargs) 
    for partite_nodes in partitions_names:
        tn_partite = {}
        if len(partite_nodes) == 0:
            #tn_partite_list.append(tn_partite)
            continue
        # construct a tn for each partite_nodes  
        partite_nodes_name = [vertex_list[t] for t in partite_nodes]
        items = tn.items()
        for (edge, vertex) in items:
            check =  all(item in partite_nodes_name for item in vertex)
            
            # determine an edge in subgraph, needs to test
            if check is True:       
                tn_partite[edge] = vertex 
                continue 
            # else: 
            #     if num_same_element(partite_nodes_name, vertex) >= 2:
            #         tn_partite[edge] = list(set(partite_nodes_name).intersection(set(vertex)))
            #         continue                   
        tn_partite_list.append(tn_partite)
        
    return tn_partite_list
  

def recur_partition(tn,**kwargs):          
    # construct a tree, grow layer by layer
    # order = []
    layer = 0;
    tn_partite_list = [[]]     
    # first partition     
    tn_partite_list[layer] = subgraph_partition(tn,**kwargs)
        
    # grow the tree
    # if the size > 1, call the partition solver 
    while max([len(x) for x in tn_partite_list[layer]])>1:
        layer += 1
        tn_partite_list.append([])
        for (count,subgraph) in enumerate(tn_partite_list[layer-1]):
            if subgraph != {}:
                tn_partite_list[layer][2*count:2*count] = subgraph_partition(subgraph,**kwargs)
       
    return tn_partite_list

def tree2order(tn_partite_list):
    tn_partite_list = tn_partite_list[::-1]
    layer_num = len(tn_partite_list) 
    order = []
    sub_opt = False # local order search for the bottom graph
    for layer in range(layer_num):
        if layer == 0:
            for subgraph in tn_partite_list[layer]:
                if sub_opt is True: 
                    # TODO: when the subgraph is small, call other order optimizor)
                    #order.append(local_search(subgraph))
                    continue
                else:
                    if subgraph != {} :                
                        result = list(subgraph.keys()) 
                        order.extend(result)
        else: #non-bottom layer, need to insert order 
            for subgraph in tn_partite_list[layer]:
                left_node_empty = 0
                if subgraph != {} :
                    node_list = list({l for word in list(subgraph.values()) for l in word}) 
                    ind_last = []
                    # find the children node of the subgraph
                    for (count_last,subgraph_last) in enumerate(tn_partite_list[layer-1]):
                        if subgraph_last != {}:
                            node_list_last = list({l for word in list(subgraph_last.values()) for l in word}) 
                            check =  all(item in node_list for item in node_list_last) 
                            if check is True:
                                ind_last.append(count_last)
                                
                    if len(ind_last) == 2:    
                        set_last1 = set(list(tn_partite_list[layer-1][ind_last[0]].keys()))
                        set_last2 = set(list(tn_partite_list[layer-1][ind_last[1]].keys()))
                        result = list(set(subgraph) - set_last1 - set_last2)
                        exist_order = [order.index(x) for x in list(set_last1)] + [order.index(x) for x in list(set_last2)]
                        ind = max(exist_order)+1
                        order[ind:ind]=result
                    
                    ### there is a single node partition in the subgraph    
                    if len(ind_last) == 1: 
                        set_last = set(list(tn_partite_list[layer-1][ind_last[0]].keys()))
                        result = list(set(subgraph) - set_last)
                        if result != [] :
                            exist_order = [order.index(x) for x in list(set_last)]
                            ind = max(exist_order)+1
                            order[ind:ind]=result
                            
                    ### TODO: maybe need to modify  
                    ### there are two single node partition in the subgraph
                    if len(ind_last) == 0:  
                        result = list(subgraph)
                        if result != [] :
                            count = tn_partite_list[layer].index(subgraph)
                            if count % 2 == 0:
                                # find the contracted edge of its paired node in the same layer
                                if len(tn_partite_list[layer][count+1].keys()) < 2:
                                    order.extend(result) 
                                    order.extend(list(tn_partite_list[layer][count+1]))
                                    # both left and right subgraphs are empty, new start
                                else:
                                    left_node_empty = 1
                                    empty_count = count
                                    left_node_buffer = result
                                    # follow the order of the right-paired subgraph   
                            else:
                                if len(tn_partite_list[layer][count-1].keys()) < 2:
                                    pass
                                    # has been added 
                                else:
                                    order[ind:ind] = result
                                    # follow the order of the left-paired subgraph
                if left_node_empty != 0:
                    exist_order = [order.index(x) for x in list(tn_partite_list[layer][empty_count+1])]
                    ind = max(exist_order)+1
                    order[ind:ind] = left_node_buffer
                       
                        
            if layer == layer_num - 1:
                set_last1 = set(list(tn_partite_list[layer][0].keys()))
                set_last2 = set(list(tn_partite_list[layer][1].keys()))  
                result = list(set(all_edge) - set_last1 - set_last2)
                if result != [] :
                    order.extend(result) 
                    
    # remove the duplicates, may happen in contracting a vertex shared by 4            
    #order2=[]
    #[order2.append(x) for x in order if x not in order2]
    assert len(order) == len(all_edge)
    return order


modes = ['direct', 'recursive']
objectives = ['cut', 'km1']  
KAHYPAR_PROFILE_DIR = join(abspath(dirname(__file__)), 'config')
#KAHYPAR_PROFILE_DIR = join(abspath(dirname(__file__)))

### generate dual graph for partition 
test_mode = 2
if test_mode == 1:
    tn ={'v_1': ['A','C'], 'v_2':['A','B'], 'v_3':['B','C','D'], 
         'v_4':['C','E'], 'v_5':['D','F'], 'v_6':['E','F']}
else:
    import networkx as nx
    N = 20
    g = nx.path_graph(N)
    comp = qtensor.DefaultQAOAComposer(g, gamma=[1], beta=[2])
    comp.ansatz_state()
    circ = comp.circuit
    tn = generate_TN.circ2tn(circ)
    # preprocessing to remove edges i_ and o_ (which have only one vertex)
    edge =list(tn.keys()); edge.sort()
    rem_num_list = [*range(N), *range(len(edge)-1, len(edge)-N-1, -1)]
    rem_list = [edge[i] for i in rem_num_list]
    [tn.pop(key) for key in rem_list]
    
all_edge = list(tn.keys())
kwargs = {'K': 2, 'eps': 0.05, 'seed': 2021, 'mode':0, 'objective':0} 
tn_partite_list = recur_partition(tn,**kwargs)        
order = tree2order(tn_partite_list)
#full_order=rem_list; full_order.extend(order)
print(order)  

### test treewidth
if test_mode != 1:
    from qtensor import utils
    old_tn=qtensor.optimisation.QtreeTensorNet.from_qtree_gates(circ)
    line_graph = old_tn.get_line_graph()
    peo = [int(x) for x in order]
    nodes, ngh = utils.get_neighbors_path(line_graph, peo)
    contraction_width = max(ngh)
    
    # from itertools import permutations
    # from random import choice
    # permutation_list = list(permutations(range(min(peo), max(peo)+1))
    # sequence = [i for i in range(len(permutation_list))]
    contraction_width_list = []; 
    import numpy as np
    np.random.seed(1)
    for _ in range(100):
        random_peo=np.random.permutation(peo)
        nodes, ngh = utils.get_neighbors_path(line_graph, list(random_peo))
        contraction_width_list.append(max(ngh))
    
    # compare with tamaki
    from qtensor.optimisation.Optimizer import TamakiTrimSlicing, GreedyOptimizer
    opt = TamakiTrimSlicing() 
    opt.max_tw = 10
    tamaki_peo, _, _ = opt.optimize(old_tn)
    tamaki_peo = tamaki_peo[2*N:len(tamaki_peo)]
    nodes, ngh = utils.get_neighbors_path(line_graph, tamaki_peo)
    tamaki_contraction_width = max(ngh)
    
    greed_opt = GreedyOptimizer()
    greedy_peo, _ = greed_opt.optimize(old_tn)
    greedy_peo = greedy_peo[2*N:len(greedy_peo)]
    nodes, ngh = utils.get_neighbors_path(line_graph, greedy_peo)
    greedy_contraction_width = max(ngh)
    print(f'Partition contraction width: {contraction_width}') 
    print(f'Min random contraction width: {min(contraction_width_list)}') 
    print(f'Tamaki contraction width: {tamaki_contraction_width}')
    print(f'Greedy contraction width: {greedy_contraction_width}')

plot_mode = 1
if (test_mode != 1 and plot_mode ==1):   
    import matplotlib.pyplot as plt  
    import qtree  
    # for line graph
    def plot_costs_peo(graph, peo, print_stat=False):
        graph, _ = qtensor.utils.reorder_graph(graph, peo)
        mems, flops = qtree.graph_model.get_contraction_costs(graph)
        qtensor.utils.plot_cost(mems, flops)
        if print_stat:
            print(f'Total FLOP: {sum(flops):e}, maximum mem: {max(mems):e}')
        return mems, flops
            
    def plot_conraction_costs(graph, peo, plot_title=''):
        f, axs = plt.subplots(1, 2, figsize=(10, 4))
        plt.sca(axs[0])
        mems, flops = plot_costs_peo(graph, peo, print_stat=True)
        plt.sca(axs[1])
        _, path = qtensor.utils.get_neighbors_path(graph, peo)
        plt.title(plot_title)
        plt.plot(path)
        plt.grid()
        print(f'Max peo: {max(path)}')
    
    # real mems & flops    
    mems, flops = qtree.graph_model.get_contraction_costs(line_graph)
    qtensor.utils.plot_cost(mems,flops)
    plot_conraction_costs(line_graph, peo)
    plt.title('Cost for partition peo')
=======
        if i in b:
            num = num + 1
    return num       
### generate dual graph for partition 
# import networkx as nx
# N = 4
# g = nx.path_graph(N)
# comp = qtensor.DefaultQAOAComposer(g, gamma=[1], beta=[2])
# comp.ansatz_state()
# circ = comp.circuit
# tn = generate_TN.circ2tn(circ)
tn ={'v_1': ['A','C'], 'v_2':['A','B'], 'v_3':['B','C','D'], 
     'v_4':['C','E'], 'v_5':['D','F'], 'v_6':['E','F']}

#graph vertex is v_1,v_2 (index), there is an edge if the index is shared
#hypergraph is the list corresponding to the vertex
#def add_edge:
    # add edge to the graph
       
#def add_vertex:
          
K = 2 
nodes, edges, hyperedge_indices, hyperedges, edge_weights, node_weights, l = ka_hg_init(tn)
hypergraph = kahypar.Hypergraph(len(nodes), len(edges), hyperedge_indices, hyperedges, K, edge_weights, node_weights)

### Set context
KAHYPAR_PROFILE_DIR = join(abspath(dirname(__file__)), 'config')
#KAHYPAR_PROFILE_DIR = join(abspath(dirname(__file__)))
modes = ['direct', 'recursive']
objectives = ['cut', 'km1']  
context = set_context(K = K, 
                       eps = 0.03,
                       mode = 1,
                       objective = 0)


### perform partition
kahypar.partition(hypergraph, context)
partitions_names = [[] for _ in range(K)]
for i, n in list(enumerate(nodes)):
    partitions_names[hypergraph.blockID(i)].append(n)
    
# construct a tree, grow layer by layer
order = []
tn_partite_list = []
for (i, partite_nodes) in enumerate(
                sorted(partitions_names)):
    tn_partite = {}
    if len(partite_nodes) == 0:
        continue
    # construct a tn for each partite_nodes  
    partite_nodes_name = [l[t] for t in partite_nodes]
    items = tn.items()
    for (node, vertex) in items:
        check =  all(item in partite_nodes_name for item in vertex)
        
        # determine a vertex in subgraph, needs to test
        if check is True:       
            tn_partite[node] = vertex 
            continue 
        else: 
            if num_same_element(partite_nodes_name, vertex) >= 2:
                tn_partite[node] = list(set(partite_nodes_name).intersection(set(vertex)))
                continue
            
    tn_partite_list.append(tn_partite)

'''
nodes, edges, hyperedge_indices, hyperedges, edge_weights, node_weights, l = ka_hg_init(tn_partite_list[0])
hypergraph2 = kahypar.Hypergraph(len(nodes), len(edges), hyperedge_indices, hyperedges, K, edge_weights, node_weights)
context = set_context(K = K, 
                       eps = 0.03,
                       mode = 1,
                       objective = 0)
kahypar.partition(hypergraph2, context)
partitions_names2 = [[] for _ in range(K)]
for i, n in list(enumerate(nodes)):
    partitions_names2[hypergraph2.blockID(i)].append(n)
'''    


    # construct a new subgraph for partite
    
    # denote the tree node as the subgraph (maybe incident matrix)
    # (could update the order after each partition)
    
    # if the size > 1, call the partition solver 
    # (or is smaller than a threshold, call other order optimizor)
>>>>>>> 10bcdad5666d7369b4cd764d756886391a73650e
