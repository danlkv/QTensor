import qtensor
from qtensor.optimisation.kahypar_ordering import generate_TN
import kahypar as kahypar
from os.path import join, abspath, dirname

def set_context(**kwargs):
        mode = modes[int(kwargs.get('mode'))]
        objective = objectives[int(kwargs.get('objective'))]
        K = int(kwargs.get('K'))
        eps = kwargs.get('eps')
        #seed = kwargs.get('seed')
        profile_mode = {'direct': 'k', 'recursive': 'r'}[mode]
        profile = f"{objective}_{profile_mode}KaHyPar_sea20.ini"
        context = kahypar.Context()
        context.loadINIconfiguration(join(KAHYPAR_PROFILE_DIR, profile))
        context.setK(K)
        #context.setSeed(seed)
        context.setEpsilon(kwargs.get('epsilon', eps * (K - 1)))
        #context.suppressOutput(kwargs.get('quiet', True))
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
