# -- Lazy import of cuquantum package
from qtensor.tools.lazy_import import LasyModule
cuquantum = LasyModule('cuquantum')
import numpy as np
cq = cuquantum # just a shorthand alias
# --

# -- Cuquantum Tensor Net

from qtensor import optimisation as qopt
from qtensor.optimisation.TensorNet import TensorNet
import string

def get_symbol(i: int) -> str:
    if i < 52:
        return string.ascii_letters[i]
    else:
        return chr(140 + i)

class CuTensorNet(TensorNet):
    def __init__(self, eq, *tdata):
        self.net = cq.Network(eq, *tdata)
        self._tensors = tdata
        self._eq = eq

    @classmethod
    def from_qtree_gates(cls, qc, init_state=None, **kwargs):
        qtr_net = qopt.QtreeTensorNet.from_qtree_gates(qc, init_state, **kwargs)
        return cls.from_qtree_tn(qtr_net)

    @staticmethod
    def get_tensor_data(tensor, data_dict):
        if hasattr(tensor, 'data'):
            if tensor.data is not None:
                return tensor.data 
        return data_dict[tensor.data_key]

    @classmethod
    def from_qtree_tn(cls, qtr_net, dtype='complex64'):
        all_data = [cls.get_tensor_data(t, qtr_net.data_dict) for t in qtr_net.tensors]
        var2char = lambda x: get_symbol(x.identity)
        all_indices = [''.join([var2char(i) for i in t.indices]) for t in qtr_net.tensors]
        eq_lh = ','.join(all_indices)
        result_ix = ''.join([var2char(i) for i in qtr_net.free_vars])
        eq = eq_lh + '->' + result_ix
        tdata = [x.astype(dtype) for x in all_data]
        return cls(eq, *tdata)
        


# -- Cuquantum Optimizer

class CuQuantumOptimizer(qopt.Optimizer):
    def __init__(self, **kwargs):
        """
        Args:
            samples (int): number of samples, scales time
            threads (int): number of threads
            slicing (cuquantum.SlicerOptions): slicing options
            reconfiguration (cuquantum.ReconfigOptions): reconfiguration options
        """
        self.kwargs = kwargs
        self.kwargs['samples'] = kwargs.get('samples', 10)
        self.kwargs['threads'] = kwargs.get('threads', 1)
        self.kwargs['slicing'] = kwargs.get('slicing', cq.SlicerOptions())
        self.kwargs['reconfiguration'] = kwargs.get('reconfiguration',
                                                    cq.ReconfigOptions(num_iterations=0))

    @property
    def peo(self):
        raise Exception("CuQuantumOptimizer does not provide elimination order."
                        " Use with compatible simulators")

    def optimize(self, tensor_net):
        path, info = tensor_net.net.contract_path(optimize=self.kwargs)
        self.treewidth = int(np.log2(info.largest_intermediate))
        return path, tensor_net

# -- Cuquantum Simulator

from qtensor import QtreeSimulator

class CuQuantumSimulator(QtreeSimulator):
    def __init__(self, optimizer=None, max_tw=None):
        self.max_tw = max_tw
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = CuQuantumOptimizer()

    def simulate_batch(self, qc, batch_vars=0, peo=None):
        self._new_circuit(qc)
        self.tn = qopt.QtreeTensorNet.from_qtree_gates(self.all_gates)
        if isinstance(batch_vars, int):
            free_final_qubits = list(range(batch_vars))
        else:
            free_final_qubits = batch_vars
        self._set_free_qubits(free_final_qubits)
        slice_dict = self._get_slice_dict()
        self.tn.slice(slice_dict)

        qtn = CuTensorNet.from_qtree_tn(self.tn)
        if peo is None:
            path, _ = self.optimizer.optimize(qtn)
            if self.max_tw:
                if self.optimizer.treewidth > self.max_tw:
                    raise ValueError(f'Treewidth {self.optimizer.treewidth} is larger than max_tw={self.max_tw}.')
        else:
            path = peo
        return cq.contract(qtn._eq, *qtn.tensors, optimize={'path': path})
