import qtree
from qensor.ProcessingFrameworks import NumpyBackend
import cirq
from qensor.optimisation.TensorNet import QtreeTensorNet
from qensor.optimisation.Optimizer import OrderingOptimizer

from loguru import logger as log

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
        tn = QtreeTensorNet(buckets, self.data_dict, self.bra_vars, self.ket_vars, fixed_vars)
        opt = OrderingOptimizer()
        peo, tn = opt.optimize(tn)
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
        log.info('batch slice {}', slice_dict)

        sliced_buckets = self.bucket_backend.get_sliced_buckets(
            self.buckets, self.data_dict, slice_dict)
        result = qtree.optimizer.bucket_elimination(
            sliced_buckets, self.bucket_backend.process_bucket,
            n_var_nosum=len(self.free_bra_vars)
        )
        print(result, result.data)
        return result.data.flatten()

    def simulate_state(self, qc, peo=None):
        return self.simulate_batch(qc, peo=peo, batch_vars=0)

    def _reorder_buckets(self):
        perm_buckets, perm_dict = qtree.optimizer.reorder_buckets(self.buckets, self.peo)
        self.ket_vars = sorted([perm_dict[idx] for idx in self.ket_vars], key=str)
        self.bra_vars = sorted([perm_dict[idx] for idx in self.bra_vars], key=str)
        self.buckets = perm_buckets
        return perm_dict

    def _get_slice_dict(self, initial_state=0, target_state=0):
        slice_dict = qtree.utils.slice_from_bits(initial_state, self.ket_vars)
        slice_dict.update(
            qtree.utils.slice_from_bits(target_state, self.bra_vars)
        )
        slice_dict.update({var: slice(None) for var in self.free_bra_vars})
        return slice_dict

class CirqSimulator(Simulator):

    def simulate(self, qc, **params):
        sim = cirq.Simulator(**params)
        return sim.simulate(qc)

