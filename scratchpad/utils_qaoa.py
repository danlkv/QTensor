import qtree
import utils
import numpy as np
import networkx as nx


def get_test_graph(S, type='grid', **kw):
    if type=='grid':
        G = nx.grid_2d_graph(S+1, (2+S)//2)
    elif type=='rectgrid':
        G = nx.grid_2d_graph(S, S)
    elif type=='rectgrid':
        G = nx.grid_2d_graph(S, S)
    elif type=='randomreg':
        n = 2*int(S*S/32)
        d = kw.get('degree', 4)
        G = nx.random_regular_graph(d, n, seed=kw.get('seed', 42))
    elif type=='randomgnp':
        n = 2*int(S*S/32)
        d = kw.get('degree', 4)
        G = nx.gnm_random_graph(n, d*n//2, seed=kw.get('seed', 42))

    #G = nx.triangular_lattice_graph(S, S)
    # remove grid labelling
    gen = (x for x in range(G.number_of_nodes()))
    G = nx.relabel_nodes(G, lambda x: next(gen))
    return G

def get_test_qaoa(S, p, type='grid', **kw):
    G = get_test_graph(S, type, **kw)
    N = G.number_of_nodes()
    beta, gamma = [np.pi/3]*p, [np.pi/2]*p
    qc = get_qaoa_circuit(G, beta, gamma)
    return qc, N

def get_test_expr_graph(S, p, type='grid', **kw):
    qc, N = get_test_qaoa(S, p, type=type, **kw)
    buck, _,_,_ = qtree.optimizer.circ2buckets(N, qc)
    graph = qtree.graph_model.buckets2graph(buck)
    return graph, N

def get_optimized_expr(S, p, **kw):
    graph, N = get_test_expr_graph(S, p, **kw)
    graph_opt, nghs = _optimise_graph(graph)
    return graph_opt, nghs, N

def cost_graph_peo(graph_old, peo):
    graph, _ = utils.reorder_graph(graph_old, peo)
    costs  = qtree.graph_model.cost_estimator(graph)
    return costs

def _optimise_graph(graph):
    peo, nghs = utils.get_locale_peo(graph, utils.n_neighbors)
    graph_opt, slice_dict = utils.reorder_graph(graph, peo)
    return graph_opt, nghs

def get_splitted_graph(S, p, pars):
    graph, N = get_test_expr_graph(S, p)
    idxs, graph = qtree.graph_model.split_graph_by_metric(graph, n_var_parallel=pars)
    graph_opt, nghs = _optimise_graph(graph)
    return graph, nghs, N

def get_cost_of_splitted(S, p, pars):
    graph, nghs, N = get_splitted_graph(S, p, pars)
    graph_opt, nghs = _optimise_graph(graph)
    mems, flops = qtree.graph_model.cost_estimator(graph_opt)
    return mems,flops,nghs, N

def get_cost_of_task(S, p=1, **kw):
    graph_opt, nghs, N = get_optimized_expr(S, p, **kw)
    mems, flops = qtree.graph_model.cost_estimator(graph_opt)
    return mems,flops,nghs, N


def simulate_circ(circuit, n_qubits, peo):
    buckets, data_dict, bra_vars, ket_vars = qtree.optimizer.circ2buckets(
        n_qubits, circuit)

    # place bra and ket variables to beginning, so these variables
    # will be contracted first
    #peo, treewidth = qtree.graph_model.get_peo(graph)

    peo = ket_vars + bra_vars + peo
    perm_buckets, perm_dict = qtree.optimizer.reorder_buckets(buckets, peo)

    # extract bra and ket variables from variable list and sort according
    # to qubit order
    ket_vars = sorted([perm_dict[idx] for idx in ket_vars], key=str)
    bra_vars = sorted([perm_dict[idx] for idx in bra_vars], key=str)

    # Take the subtensor corresponding to the initial state
    initial_state = target_state = 0
    slice_dict = qtree.utils.slice_from_bits(initial_state, ket_vars)
    slice_dict.update(
        qtree.utils.slice_from_bits(target_state, bra_vars)
    )
    sliced_buckets = qtree.np_framework.get_sliced_np_buckets(
        perm_buckets, data_dict, slice_dict)
    
    result = qtree.optimizer.bucket_elimination(
        sliced_buckets, qtree.np_framework.process_bucket_np)
    return result

def layer_of_Hadamards(qc,N):
    layer = []
    for q in range(N):
        layer.append(qtree.operators.H(q))
    qc.append(layer)

def get_qaoa_circuit(G, beta, gamma):
    assert(len(beta) == len(gamma))
    p = len(beta) # infering number of QAOA steps from the parameters passed
    N = G.number_of_nodes()
    qc = []
    layer_of_Hadamards(qc, N)
    # second, apply p alternating operators
    for i in range(p):
        qc += get_cost_operator_circuit(G,gamma[i])
        qc += get_mixer_operator_circuit(G,beta[i])
    # finally, do not forget to measure the result!
    return qc

def append_x_term(qc, q1, beta):
    layer = []
    layer.append(qtree.operators.H(q1))
    layer.append(qtree.operators.ZPhase(q1, alpha=2*beta))
    layer.append(qtree.operators.H(q1))
    qc.append(layer)

def get_mixer_operator_circuit(G, beta):
    N = G.number_of_nodes()
    qc = []
    for n in G.nodes():
        append_x_term(qc, n, beta)
    return qc

def append_zz_term(qc, q1, q2, gamma):
    layer = []
    layer.append(qtree.operators.cX(q1, q2))
    layer.append(qtree.operators.ZPhase(q2, alpha=2*gamma))
    layer.append(qtree.operators.cX(q1, q2))
    qc.append(layer)

def get_cost_operator_circuit(G, gamma):
    N = G.number_of_nodes()
    qc = list()
    for i, j in G.edges():
        append_zz_term(qc, i, j, gamma)
    return qc
