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

    def optimize_buckets(self, buckets, ignored_vars=[], fixed_vars: list=None):
        graph = qtree.graph_model.buckets2graph(buckets,
                                               ignore_variables=ignored_vars)
        if fixed_vars:
            graph = qtree.graph_model.make_clique_on(graph, fixed_vars)

        peo_ints, step_nghs = utils.get_locale_peo(graph, utils.n_neighbors)
        self.treewidth = max(step_nghs)

        if fixed_vars:
            peo = qtree.graph_model.get_equivalent_peo(graph, peo_ints, fixed_vars)

        peo = [qtree.optimizer.Var(var, size=graph.nodes[var]['size'],
                        name=graph.nodes[var]['name'])
                    for var in peo_ints]

        peo = ignored_vars + peo
        self.peo = peo
        return peo

    def _new_circuit(self, qc):
        all_gates = qc
        n_qubits = len(set(sum([g.qubits for g in all_gates], tuple())))
        self.n_qubits = n_qubits
        self.qtree_circuit = [[g] for g in qc]

    def _create_buckets(self):
        buckets, data_dict, bra_vars, ket_vars = qtree.optimizer.circ2buckets(
            self.n_qubits, self.qtree_circuit)
        self.buckets = buckets
        self.data_dict = data_dict
        self.bra_vars = bra_vars
        self.ket_vars = ket_vars

    def _set_free_qubits(self, free_final_qubits):
        self.free_bra_vars = [self.bra_vars[i] for i in free_final_qubits]
        self.bra_vars = [var for var in self.bra_vars if var not in self.free_bra_vars]

    def _optimize_buckets(self):
        self.peo = self.optimize_buckets(
            self.buckets, ignored_vars=self.bra_vars+self.ket_vars, fixed_vars=self.free_bra_vars)

    def _reorder_buckets(self):
        perm_buckets, perm_dict = qtree.optimizer.reorder_buckets(self.buckets, self.peo)
        self.ket_vars = sorted([perm_dict[idx] for idx in self.ket_vars], key=str)
        self.bra_vars = sorted([perm_dict[idx] for idx in self.bra_vars], key=str)
        self.buckets = perm_buckets

    def _get_slice_dict(self, initial_state=0, target_state=0):
        slice_dict = qtree.utils.slice_from_bits(initial_state, self.ket_vars)
        slice_dict.update(
            qtree.utils.slice_from_bits(target_state, self.bra_vars)
        )
        slice_dict.update({var: slice(None) for var in self.free_bra_vars})
        return slice_dict

    def simulate_batch(self, qc, batch_vars=0, peo=None):
        self._new_circuit(qc)
        self._create_buckets()
        # Collect free qubit variables
        free_final_qubits = list(range(batch_vars))
        self._set_free_qubits(free_final_qubits)
        if peo is None:
            self._optimize_buckets()
        else:
            self.peo = peo

        self._reorder_buckets()
        slice_dict = self._get_slice_dict()

        sliced_buckets = self.bucket_backend.get_sliced_buckets(
            self.buckets, self.data_dict, slice_dict)
        result = qtree.optimizer.bucket_elimination(
            sliced_buckets, self.bucket_backend.process_bucket)
        return result.data.flatten()

    def simulate_state(self, qc, peo=None):
        return self.simulate_batch(qc, peo=peo, batch_vars=0)





    def simulate_batch_old(self, qc, peo=None, batch_vars=1):
        all_gates = qc
        n_qubits = len(set(sum([g.qubits for g in all_gates], tuple())))
        self.n_qubits = n_qubits
        circuit = [[g] for g in qc]

        # Collect free qubit variables
        free_final_qubits = list(range(batch_vars))

        buckets, data_dict, bra_vars, ket_vars = qtree.optimizer.circ2buckets(
            n_qubits, circuit)

        assert len(free_final_qubits)<=self.n_qubits, 'Batch size should be no larger than n_qubits'

        free_bra_vars = [bra_vars[i] for i in free_final_qubits]
        bra_vars = [var for var in bra_vars if var not in free_bra_vars]

        if peo is None:
            peo = self.optimize_buckets(buckets, ignored_vars=bra_vars+ket_vars, fixed_vars=free_bra_vars)

        return self._slice_simulate(buckets, peo, data_dict, bra_vars, ket_vars, free_bra_vars)


    def _slice_simulate(self, buckets, peo, data_dict, bra_vars, ket_vars, free_bra_vars):
        perm_buckets, perm_dict = qtree.optimizer.reorder_buckets(buckets, peo)
        ket_vars = sorted([perm_dict[idx] for idx in ket_vars], key=str)
        bra_vars = sorted([perm_dict[idx] for idx in bra_vars], key=str)

        initial_state = target_state = 0
        slice_dict = qtree.utils.slice_from_bits(initial_state, ket_vars)
        slice_dict.update(
            qtree.utils.slice_from_bits(target_state, bra_vars)
        )
        slice_dict.update({var: slice(None) for var in free_bra_vars})

        sliced_buckets = self.bucket_backend.get_sliced_buckets(
            perm_buckets, data_dict, slice_dict)
        result = qtree.optimizer.bucket_elimination(
            sliced_buckets, self.bucket_backend.process_bucket)
        return result.data.flatten()

class CirqSimulator(Simulator):

    def simulate(self, qc, **params):
        sim = cirq.Simulator(**params)
        return sim.simulate(qc)

