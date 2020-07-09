import numpy as np
import networkx as nx
from functools import partial
from qiskit.optimization.ising.max_cut import get_operator as get_maxcut_operator
from variationaltoolkit.objectivewrapper import ObjectiveWrapper
from variationaltoolkit.objectives import maxcut_obj


if __name__ == "__main__":
    elist = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 5], [1, 6], [2, 7], [3, 8], [4, 9], [5, 7], [5, 8], [6, 8], [6, 9], [7, 9]]
    G = nx.OrderedGraph()
    G.add_edges_from(elist)
    w = nx.adjacency_matrix(G, nodelist=range(10)).toarray()
    obj = partial(maxcut_obj,w=w)
    C, _ = get_maxcut_operator(w)
    obj_sv = ObjectiveWrapper(obj, 
            varform_description={'name':'QAOA', 'p':9, 'num_qubits':10, 'cost_operator':C}, 
            backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'}, 
            objective_parameters={'save_resstrs':True},
            execute_parameters={})
    parameters = np.array([5.192253984583296, 5.144373231492732, 5.9438949617723775, 5.807748946652058, 3.533458907810596, 6.006206583282401, 6.122313961527631, 6.218468942101044, 6.227704753217614, 0.3895570099244132, -0.1809282325810937, 0.8844522327007089, 0.7916086532373585, 0.21294534589417236, 0.4328896243354414, 0.8327451563500539, 0.7694639329585451, 0.4727893829336214])
    print(f"QAOA objective {obj_sv.get_obj()(parameters)} has to be close to optimal cut -12")
