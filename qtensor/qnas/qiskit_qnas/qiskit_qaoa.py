import networkx as nx
import argparse
import numpy as np
from qiskit import Aer
from qiskit.algorithms.minimum_eigen_solvers.qaoa import QAOAAnsatz
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import GoemansWilliamsonOptimizer
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import TwoLocal


ROT_OPS =  ['rx', 'ry', 'rz', 'h', 'x', 'y',
           'z']

ENT_OPS = ['cx', 'cy', 'cz', 'crx', 'cry', 'crz', 'rzz', 
           'rxx', 'ryy']

def get_graphs(graphs_file, energies_file):
    # graphs_file = open("QAOA_Dataset/20_10_node_erdos_renyi_graphs.txt")
    matrix_list = np.loadtxt(graphs_file).reshape(20, 10, 10)
    graph_list = matrices_to_graphs(matrix_list)
    energies = [int(l.strip()) for l in open(energies_file, 'r').readlines()]
    return graph_list, energies


def matrices_to_graphs(matrix_list):
    g_list = []
    for matrix in matrix_list:
        array = np.array(matrix)
        g = nx.from_numpy_array(array)
        g_list.append(g)

    return g_list


def get_adj_mat(G):
    n = G.number_of_nodes()
    adj = np.zeros((n, n))
    for e in G.edges:
        if nx.is_weighted(G):
            adj[e[0]][e[1]] = G[e[0]][e[1]]['weight']
            adj[e[1]][e[0]] = G[e[0]][e[1]]['weight']
        else:
            adj[e[0]][e[1]] = 1
            adj[e[1]][e[0]] = 1
    
    return adj


def get_maxcut_problem(G):
    adj = get_adj_mat(G)
    n_qubits = G.number_of_nodes()
    problem = QuadraticProgram()
    prob = [problem.binary_var(f"x{i}") for i in range(n_qubits)]
    problem.maximize(linear=adj.dot(np.ones(n_qubits)), 
                     quadratic=-adj)
    return problem



def add_x_gate(qc, qubit,  beta):
    qc.rx(2*beta, qubit)

def add_y_gate(qc, qubit, beta):
    qc.ry(2*beta, qubit)

def add_xx_gate(qc, qubit, beta):
    qc.rx(2*beta, qubit)
    qc.rx(2*beta, qubit)


def add_yy_gate(qc,qubit, beta):
    qc.ry(2*beta, qubit)
    qc.ry(2*beta, qubit)

def add_zz_term(qc, qi, qj, gamma):
    qc.cx(qi, qj)
    qc.rz(2 * gamma, qj)
    qc.cx(qi, qj)

def build_cost_circuit(G, gamma):
    N = G.number_of_nodes()
    qc = QuantumCircuit(N)
    for i, j in G.edges():
        if nx.is_weighted(G):
            add_zz_term(qc, i, j, gamma * G[i][j]["weight"])
        else:
            add_zz_term(qc, i, j, gamma)
    return qc

# def build_mixer_circuit(G, beta, gate_str):
#     N = G.number_of_nodes()
#     qc = QuantumCircuit(N)
#     for n in G.nodes():
#         if gate_str == 'x':
#             add_x_gate(qc, n, beta)
#         elif gate_str == 'y':
#             add_y_gate(qc, n, beta)
#         elif gate_str == 'xx':
#             add_xx_gate(qc, n, beta)
#         elif gate_str == 'yy':
#             add_yy_gate(qc, n, beta)
#         else:
#             raise ValueError(f"Invalid mixer str {gate_str}")
    
#     return qc

def build_mixer_circuit(G, args):
    N = G.number_of_nodes()
    rot_gates = np.random.choice(ROT_OPS, args.nrots).tolist()
    ent_gates = np.random.choice(ENT_OPS, args.nents).tolist()

    qc = TwoLocal(num_qubits=N, rotation_blocks=rot_gates, entanglement_blocks=ent_gates,
                reps=args.reps, entanglement='reverse_linear')
    
    # qc = qc.assign_parameters(beta)
    return qc


def get_maxcut_qaoa_ckt(G, beta, gamma, args):
    assert len(beta) == len(gamma)
    p = len(beta)
    N = G.number_of_nodes()
    qr = QuantumRegister(N)
    qc = QuantumCircuit(qr)

    qc.h(range(N))
    for i in range(p):
        qc = qc.compose(build_cost_circuit(G, gamma[i]))
        mixer = build_mixer_circuit(G, args).decompose()
        mixer_qc = mixer.assign_parameters(np.repeat(beta[i], repeats=mixer.num_parameters))
        qc = qc.compose(mixer_qc)

        # qc = qc.compose(build_mixer_circuit(G, beta[i],mixer_str))

    return qc


if __name__ == '__main__':
    simulator = Aer.get_backend('aer_simulator')
    graphs, energies = graphs, energies = get_graphs('QAOA_Dataset/20_10_node_erdos_renyi_graphs.txt',  
                                                 'QAOA_Dataset/20_10_node_erdos_renyi_graphs_energies.txt')

    p = 1
    beta = [np.random.random() for _ in range(p)]
    gamma = [np.random.random() for _ in range(p)]

    args = argparse.Namespace()
    args.nrots = 2
    args.nents = 2
    args.reps = 2

    qc = get_maxcut_qaoa_ckt(graphs[0], gamma=gamma, beta=beta, args=args)
    qc.measure_all()
    counts = simulator.run(qc).result().get_counts()
    
# def make_qaoa_circuit(G, p, angles, mixer=None):
#     problem = get_maxcut_problem(G)
#     C, offset = problem.to_ising()
#     if not mixer:
#         ansatz = QAOAAnsatz(C, p).decompose()
#     else:
#         mixer = mixer if isinstance(mixer, QuantumCircuit) else \
#         get_mixer_ckt(G, mixer)
#         ansatz = QAOAAnsatz(C, p, mixer_operator=mixer).decompose()
#         # mix_qc = get_mixer_ansatz(G, mixer_str)
#         # ansatz = QAOAAnsatz(C, p, mixer_operator=mix_qc).decompose()
    
#     qc = ansatz.bind_parameters(angles)
#     return qc, C, offset