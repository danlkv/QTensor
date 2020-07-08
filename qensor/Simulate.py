import qtree
from qensor.ProcessingFrameworks import NumpyBackend
import cirq

from qensor import utils

class Simulator:
    def __init__(self):
        pass

    def simulate(self, qc):
       """ Factory method """
       raise NotImplementedError()


class QtreeSimulator(Simulator):
    def __init__(self, bucket_backend=NumpyBackend()):
        self.bucket_backend = bucket_backend

    def simulate(self, qc):
        return self.simulate_state(qc)

    def optimize_buckets(self, buckets, ignored_vars=[]):
        graph = qtree.graph_model.buckets2graph(buckets,
                                               ignore_variables=ignored_vars)

        peo_ints, treewidth = utils.get_locale_peo(graph, utils.n_neighbors)

        peo = [qtree.optimizer.Var(var, size=graph.nodes[var]['size'],
                        name=graph.nodes[var]['name'])
                    for var in peo_ints]

        peo = ignored_vars + peo
        self.peo = peo
        return peo

    def simulate_state(self, qc, peo=None):
        all_gates = qc
        n_qubits = len(set(sum([g.qubits for g in all_gates], tuple())))
        self.n_qubits = n_qubits
        circuit = [[g] for g in qc]


        buckets, data_dict, bra_vars, ket_vars = qtree.optimizer.circ2buckets(
            n_qubits, circuit)
        
        if peo is None:
            peo = self.optimize_buckets(buckets, ignored_vars=bra_vars+ket_vars)

        perm_buckets, perm_dict = qtree.optimizer.reorder_buckets(buckets, peo)
        ket_vars = sorted([perm_dict[idx] for idx in ket_vars], key=str)
        bra_vars = sorted([perm_dict[idx] for idx in bra_vars], key=str)

        initial_state = target_state = 0
        slice_dict = qtree.utils.slice_from_bits(initial_state, ket_vars)
        slice_dict.update(
            qtree.utils.slice_from_bits(target_state, bra_vars)
        )
        sliced_buckets = self.bucket_backend.get_sliced_buckets(
            perm_buckets, data_dict, slice_dict)
        result = qtree.optimizer.bucket_elimination(
            sliced_buckets, self.bucket_backend.process_bucket)
        return result

class CirqSimulator(Simulator):

    def simulate(self, qc, **params):
        sim = cirq.Simulator(**params)
        return sim.simulate(qc)

