"""
This module implements interface to KaHypar program.
"""
#import qtensor
#from qtensor.optimisation.kahypar_ordering import generate_TN
import kahypar as kahypar
from os.path import join, abspath, dirname

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

modes = ['direct', 'recursive']
objectives = ['cut', 'km1']  
KAHYPAR_PROFILE_DIR = join(abspath(dirname(__file__)), 'config')
#KAHYPAR_PROFILE_DIR = join(abspath(dirname(__file__)))

def set_context(**kwargs):
        mode = modes[int(kwargs.get('mode'))]
        objective = objectives[int(kwargs.get('objective'))]
        K = int(kwargs.get('K'))
        eps = kwargs.get('eps')
        seed = kwargs.get('seed')
        profile_mode = {'direct': 'k', 'recursive': 'r'}[mode]
        profile = f"{objective}_{profile_mode}KaHyPar_sea20.ini"
        context = kahypar.Context()
        context.loadINIconfiguration(join(KAHYPAR_PROFILE_DIR, profile))
        context.setK(K)
        context.setSeed(seed)
        context.setEpsilon(kwargs.get('epsilon', eps * (K - 1))) #imbalance
        context.suppressOutput(True)
        return context

def ka_hg_init(tn): # tn: a dictionary from circ2tn
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

def single_partition(tn,**kwargs): 
    K = int(kwargs.get('K'))
    nodes, edges, hyperedge_indices, hyperedges, edge_weights, node_weights, l = ka_hg_init(tn)
    hypergraph = kahypar.Hypergraph(len(nodes), len(edges), hyperedge_indices, \
                                    hyperedges, K, edge_weights, node_weights)
    
    ### Set context  
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
                 
        tn_partite_list.append(tn_partite)
        
    return tn_partite_list
  

def recur_partition(tn,**kwargs):          
    # construct a tree, grow layer by layer
    layer = 0;
    tn_partite_list = [[]]     
    # first partition     
    tn_partite_list[layer] = subgraph_partition(tn,**kwargs)
        
    # grow the tree
    # if the size > K-1, call the partition solver 
    K = int(kwargs.get('K'))
    while max([len(x) for x in tn_partite_list[layer]]) > K-1:
        layer += 1
        tn_partite_list.append([])
        for (count,subgraph) in enumerate(tn_partite_list[layer-1]):
            if subgraph != {}: # important
                tn_partite_list[layer][2*count:2*count] = subgraph_partition(subgraph,**kwargs)
       
        # TODO: Adjust the hyperparameters during the partition
        
    return tn_partite_list


def find_child_ind(subgraph,tn_partite_list, layer):
    # child index is a list  
    ind_last = []
    if subgraph != {} : 
        node_list = list({l for word in list(subgraph.values()) for l in word}) 
        for (count_last,subgraph_last) in enumerate(tn_partite_list[layer+1]):
                    if subgraph_last != {}:
                        node_list_last = list({l for word in list(subgraph_last.values()) for l in word}) 
                        check =  all(item in node_list for item in node_list_last) 
                        if check is True:
                            ind_last.append(count_last)                           
    return ind_last

def find_parent_ind(subgraph,tn_partite_list, layer):
    # parent index is a scalar (int) 
    if subgraph != {} : 
        node_list = list({l for word in list(subgraph.values()) for l in word}) 
        for (count_last,subgraph_last) in enumerate(tn_partite_list[layer-1]):
                    if subgraph_last != {}:
                        node_list_last = list({l for word in list(subgraph_last.values()) for l in word}) 
                        check =  all(item in node_list_last for item in node_list) 
                        if check is True:
                            ind_last=count_last                           
    return ind_last
    
    
def tree2order(tn,tn_partite_list):
    # find the order from top to bottom, arbitary K
    all_edge = list(tn.keys())
    layer_num = len(tn_partite_list) 
    order = []
    import copy
    order_tree = copy.deepcopy(tn_partite_list)
    t = 0 # count the temp result in order_tree               
    for layer in range(layer_num):
        if layer == 0: 
            # top layer   
            set_last = []
            for subgraph in tn_partite_list[layer]:
                set_last.extend(list(subgraph.keys()))
            result = list(set(all_edge) - set(set_last))
            if result == [] :
                result = [f'temp_{t}']
                t = t + 1
            order.extend(result) 
                
            # layer0 and layer1          
            for (count,subgraph) in enumerate(tn_partite_list[layer]):
                ind_last = find_child_ind(subgraph,tn_partite_list, layer)
                set_last = []
                [set_last.extend(list(tn_partite_list[layer+1][ind].keys())) for ind in ind_last]  
                result = list(set(subgraph) - set(set_last))
                if result == []:
                    result = [f'temp_{t}']
                    t = t + 1
                order[0:0] = result
                order_tree[layer][count] = result
        else:                        
            #Middle layers, need to insert order 
            for (count,subgraph) in enumerate(tn_partite_list[layer]):
                if subgraph != {} : 
                    ind_last = []
                    if layer != layer_num - 1:
                        ind_last = find_child_ind(subgraph,tn_partite_list, layer)
                                               
                    set_last = []
                    [set_last.extend(list(tn_partite_list[layer+1][ind].keys())) for ind in ind_last]  
                    result = list(set(subgraph) - set(set_last))
                    if result == []:
                        result = [f'temp_{t}']
                        t = t + 1
                    partent_ind = find_parent_ind(subgraph,tn_partite_list, layer)
                    parent_set = list(order_tree[layer-1][partent_ind])
                    exist_order = [order.index(x) for x in list(parent_set) if x in order]
                    ind = min(exist_order)
                    order[ind:ind] = result
                    order_tree[layer][count] = result                        
                      
    order = [x for x in order if type(x) != str]
    assert len(order) == len(all_edge)
    # complete the top of order_tree
    set_last1 = set(list(tn_partite_list[0][0].keys()))
    set_last2 = set(list(tn_partite_list[0][1].keys()))  
    result = list(set(all_edge) - set_last1 - set_last2)
    order_tree= [result] + order_tree
    return order,order_tree
