import sys
import networkx as nx
import qtree

sys.path.append('..')
import utils_qaoa
import utils

class Simulation:
    def __init__(self, qc):
        self.qc = qc

    def generate_tensor_net(self):
        graph = nx.Graph()
        qubit_map = {}
        for gate in self.circuit_gates():
            graph.add_node(gate)
            other_gates = []
            for i, qubit in enumerate(self.iter_qubits(gate)):
                prev_gate = qubit_map.get(qubit)
                if prev_gate:
                    other_gates.append(prev_gate)

                qubit_map[qubit] = gate
                graph.add_edges_from([ gate, prev_gate ])

        return graph

    def all_qubits(self):
       qubits = []
       for gate in self.circuit_gates():
           for qubit in self.iter_qubits(gate):
               qubits.append(qubit)
       return set(qubits)

    @property
    def n_qubits(self):
        return len(self.all_qubits())

    def iter_qubits(self, gate):
       """ Factory method """
       for qubit in gate.qubits:
            yield qubit

    def circuit_gates(self):
       """ Factory method """
       for gate in self.qc:
            yield gate

    def simulate(self):
       """ Factory method """
       raise NotImplementedError()


class QtreeSimulate(Simulation):

    def iter_qubits(self, gate):
        return super().iter_qubits(gate)

    def circuit_gates(self):
        # Flatten the thing
        all_gates = sum(self.qc, [])
        return all_gates

    def simulate(self):
        return self.simulate_state()

    def simulate_state(self):
        n_qubits = self.n_qubits
        circuit = self.qc

        buckets, data_dict, bra_vars, ket_vars = qtree.optimizer.circ2buckets(
            n_qubits, circuit)

        graph = qtree.graph_model.buckets2graph(buckets,
                                               ignore_variables=ket_vars+bra_vars)

        peo_ints, treewidth = utils.get_locale_peo(graph, utils.n_neighbors)

        peo = [qtree.optimizer.Var(var, size=graph.nodes[var]['size'],
                        name=graph.nodes[var]['name'])
                    for var in peo_ints]

        peo = ket_vars + bra_vars + peo
        perm_buckets, perm_dict = qtree.optimizer.reorder_buckets(buckets, peo)
        ket_vars = sorted([perm_dict[idx] for idx in ket_vars], key=str)
        bra_vars = sorted([perm_dict[idx] for idx in bra_vars], key=str)

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
