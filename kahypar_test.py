import os
import kahypar as kahypar

num_nodes = 7
num_nets = 4

hyperedge_indices = [0,2,6,9,12]
hyperedges = [0,2,0,1,3,4,3,4,6,2,5,6]

node_weights = [1,2,3,4,5,6,7]
edge_weights = [11,22,33,44]

k=2

hypergraph = kahypar.Hypergraph(num_nodes, num_nets, hyperedge_indices, hyperedges, k, edge_weights, node_weights)

context = kahypar.Context()
context.loadINIconfiguration("/Users/filipmazurek/Documents/Simulator_Argonne/kahypar/config/cut_kKaHyPar_sea20.ini")

context.setK(k)
context.setEpsilon(0.03)

kahypar.partition(hypergraph, context)
print("success")
