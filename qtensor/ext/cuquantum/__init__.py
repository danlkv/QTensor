# -- Lazy import of cuquantum package
from qtensor.tools.lazy_import import LasyModule
cuquantum = LasyModule('cuquantum')
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

    @classmethod
    def from_qtree_tn(cls, qtr_net, dtype='complex64'):
        all_data = [qtr_net.data_dict[t.data_key] for t in qtr_net.tensors]
        var2char = lambda x: get_symbol(x.identity)
        all_indices = [''.join([var2char(i) for i in t.indices]) for t in qtr_net.tensors]
        eq_lh = ','.join(all_indices)
        result_ix = ''.join([var2char(i) for i in qtr_net.free_vars])
        eq = eq_lh + '->' + result_ix
        tdata = [x.astype(dtype) for x in all_data]
        return cls(eq, *tdata)
        

