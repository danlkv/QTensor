import numpy as np
import scipy
import itertools
from functools import lru_cache

from qtensor.tests import get_test_problem
from qtensor import GenIsing

@lru_cache(maxsize=2**12)
def get_gen_ising_test_problem(n=10, p=2, d=3, type='random'):
    """
    Get test QAOA problem

    Args:
        n: number of nodes in graph
        p: number of qaoa cycles
        d: degree of graph
        type: type of graph

    Returns
        (nx.Graph, gamma, beta)
    """
    G, gamma, beta = get_test_problem(n, p, d, type)
    
    offset_range = [-100, 100] #constant term
    node_weight_range = [-5, 5] #linear terms
    edge_weight_range = [-2, 2] #quadratic terms
    
    rng = np.random.default_rng()
    
    G.graph['offset'] = rng.uniform(*offset_range)
    for node in sorted(G.nodes()):
        G.nodes[node]['weight'] = rng.uniform(*node_weight_range)
    for edge in sorted(G.edges()):
        G.edges[edge]['weight'] = rng.uniform(*edge_weight_range)
    
    return G, gamma, beta

class scipyIsingQAOASimulator:
    def __init__(self, G):
        self.G = G
        
        self.n_var = G.number_of_nodes()
        self.n_dim = 2**self.n_var
        self.var_names = sorted(G.nodes())
        
        self.s_array = np.full((self.n_dim, 1), self.n_dim**-.5)
        
        ## B matrix ##########
        self.B_array = np.zeros((self.n_dim, self.n_dim))
        for i in range(self.n_var):
            self.B_array[2**i:2**(i+1), 0:2**i] = np.identity(2**i)
            self.B_array[0:2**i, 2**i:2**(i+1)] = np.identity(2**i)
            self.B_array[2**i:2**(i+1), 2**i:2**(i+1)] = \
                                    self.B_array[0:2**i, 0:2**i]
        ######################
        
        ## C matrix ##########
        self.C_eigenvalues = []
        for substitution_vals in itertools.product([1, -1], repeat=self.n_var):
            substitution_dict = dict(zip(self.var_names, substitution_vals))
            self.C_eigenvalues.append(
                self.eigen_energy(substitution_dict)
            )
        self.C_array = np.diag(self.C_eigenvalues)
        ######################
        
        self.p1_energy_expectation_vectorized = np.vectorize(self._p1_energy_expectation)
    
    def eigen_energy(self, substitution_dict):
        energy = 0
        if 'offset' in self.G.graph:
            energy += self.G.graph['offset']
        
        for i, w in self.G.nodes.data('weight'):
            if w is not None:
                energy += w*substitution_dict[i]
        
        for i, j, w in self.G.edges.data('weight'):
            if w is not None:
                energy += w*substitution_dict[i]*substitution_dict[j]
        
        return energy
    
    def _state_ket_to_energy(self, state_ket):
        state_bra = np.conj(state_ket.T)
        
        tmp = np.tensordot(self.C_array, state_ket, axes=1)
        return np.tensordot(state_bra, tmp, axes=1).flatten()[0]
    
    def _p1_energy_expectation(self, gamma, beta):
        state_ket = self.s_array
        
        operator = scipy.linalg.expm(-1j*gamma*self.C_array)
        state_ket = np.tensordot(operator, state_ket, axes=1)
        
        operator = scipy.linalg.expm(-1j*beta*self.B_array)
        state_ket = np.tensordot(operator, state_ket, axes=1)
        
        return self._state_ket_to_energy(state_ket)
    
    def energy_expectation(self, gamma, beta):
        assert len(gamma) == len(beta)
        
        state_ket = self.s_array
        for i in range(len(gamma)):
            operator = scipy.linalg.expm(-1j*gamma[i]*self.C_array)
            state_ket = np.tensordot(operator, state_ket, axes=1)
            
            operator = scipy.linalg.expm(-1j*beta[i]*self.B_array)
            state_ket = np.tensordot(operator, state_ket, axes=1)
        
        return self._state_ket_to_energy(state_ket)

def test_graph_qubo_converters():
    G, _, _ = get_gen_ising_test_problem(n=16, p=2, d=3)
    
    mdl = GenIsing.graph_to_docplexqubo(G)
    qp = GenIsing.graph_to_qiskitqubo(G)
    
    G1 = GenIsing.docplexqubo_to_graph(mdl)
    G2 = GenIsing.qiskitqubo_to_graph(qp)
    
    assert mdl and qp and G1 and G2

def test_gen_ising_energy():
    G, gamma, beta = get_gen_ising_test_problem(n=16, p=2, d=3)
    sim = GenIsing.QtreeIsingQAOASimulator(GenIsing.QtreeIsingQAOAComposer)
    
    E = sim.energy_expectation(
        G=G, gamma=gamma, beta=beta)[0]
    
    print(f"Energy = {E}")
    assert np.abs(np.imag(E))<1e-6
    
    for (n, p, d) in [(6, 2, 0), (6, 2, 3), (6, 2, 5)]:
        G, gamma, beta = get_gen_ising_test_problem(n=n, p=p, d=d)
        qtensor_sim = GenIsing.QtreeIsingQAOASimulator(GenIsing.QtreeIsingQAOAComposer)
        qtensor_energy = qtensor_sim.energy_expectation(
            G=G, gamma=gamma, beta=beta
        )[0]
        
        pi_gamma = [np.pi*_ for _ in gamma]
        pi_beta = [np.pi*_ for _ in beta]
        
        scipy_sim = scipyIsingQAOASimulator(G)
        scipy_energy = scipy_sim.energy_expectation(pi_gamma, pi_beta)
        
        print(f"{qtensor_energy = }, {scipy_energy = }")
        assert np.abs(np.imag(qtensor_energy)) < 1e-6
        assert np.abs(np.imag(qtensor_energy)) < 1e-6
        assert np.isclose(
            np.real(qtensor_energy), np.real(scipy_energy),
            rtol=1e-4, atol=1e-7
        )

if __name__ == "__main__":
    test_graph_qubo_converters()
    test_gen_ising_energy()