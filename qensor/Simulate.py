import qtree
from qensor.ProcessingFrameworks import NumpyBackend
import cirq
from qensor.optimisation.TensorNet import QtreeTensorNet
from qensor.optimisation.Optimizer import OrderingOptimizer



from loguru import logger as log

from qensor import utils

def int_slice(value, vars_to_slice):
    """
    Creates a slice dict with integers an values.
    """
    dimensions = [var.size for var in vars_to_slice]
    multiindex = qtree.utils.unravel_index(value, dimensions)

    return {idx: val for idx, val in zip(vars_to_slice, multiindex)}

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

    def _create_buckets(self):
        self.tn = QtreeTensorNet.from_qtree_gates(self.all_gates)

    def _new_circuit(self, qc):
        self.all_gates = qc

    def optimize_buckets(self):
        opt = OrderingOptimizer()
        peo, self.tn = opt.optimize(self.tn)
        return peo

    def _set_free_qubits(self, free_final_qubits):
        self.tn.free_vars = [self.tn.bra_vars[i] for i in free_final_qubits]
        self.tn.bra_vars = [var for var in self.tn.bra_vars if var not in self.tn.free_vars]

    def _optimize_buckets(self):
        self.peo = self.optimize_buckets()

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
        #log.info('batch slice {}', slice_dict)

        sliced_buckets = self.tn.slice(slice_dict)

        result = qtree.optimizer.bucket_elimination(
            sliced_buckets, self.bucket_backend.process_bucket,
            n_var_nosum=len(self.tn.free_vars)
        )
        #print(result, result.data)
        return result.data.flatten()

    def simulate_state(self, qc, peo=None):
        return self.simulate_batch(qc, peo=peo, batch_vars=0)

    def _reorder_buckets(self):
        perm_buckets, perm_dict = qtree.optimizer.reorder_buckets(self.tn.buckets, self.peo)
        self.tn.ket_vars = sorted([perm_dict[idx] for idx in self.tn.ket_vars], key=str)
        self.tn.bra_vars = sorted([perm_dict[idx] for idx in self.tn.bra_vars], key=str)
        self.tn.buckets = perm_buckets
        return perm_dict

    def _get_slice_dict(self, initial_state=0, target_state=0):
        slice_dict = int_slice(initial_state, self.tn.ket_vars)
        slice_dict.update(int_slice(target_state, self.tn.bra_vars))
        slice_dict.update({var: slice(None) for var in self.tn.free_vars})
        return slice_dict

class CirqSimulator(Simulator):

    def simulate(self, qc, **params):
        sim = cirq.Simulator(**params)
        return sim.simulate(qc)

