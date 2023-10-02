import numpy as np
import networkx as nx
from loguru import logger
import qtree, qtensor
import sys
from typing import Iterable, Union

from qtensor.optimisation.TensorNet import circ2buckets_init

logger.remove()
logger.add(sys.stderr, level='DEBUG')

class ContractionInfo:
    pass

class TNAdapter:
    def __init__(self, *args, **kwargs):
        pass

    def optimize(self, out_indices: Iterable = []) -> ContractionInfo:
        return ContractionInfo()
    
    # Inplace or not?
    def slice(self, slice_dict: dict) -> 'TNAdapter':
        ...

    # Inplace or not?
    def contract(self, contraction_info: ContractionInfo) -> np.ndarray:
        ...

    # require this?
    def copy(self):
        pass

    def add_tensor(self, data, indices):
        pass

def add_random_tn(tn:TNAdapter, graph:nx.Graph, e2i:dict={}, min_ix=0, dim=2):
    e2i_default = {tuple(sorted(e)):i for i,e in enumerate(graph.edges(), min_ix)}
    if set(e2i.keys()).intersection(set(e2i_default.keys())):
        raise ValueError("e2i and e2i_default have common keys")
    # overwrite default with e2i
    e2i_default.update(e2i)
    e2i = e2i_default
    logger.debug("Indices: {}", e2i)
    for u in graph:
        indices = []
        for v in graph[u]:
            edge = tuple(sorted((u,v)))
            indices.append(e2i[edge])
        #tn.add_tensor(np.random.randn(*[dim]*len(indices)), indices)
        tn.add_tensor(1+np.random.rand(*[dim]*len(indices)), indices)
    return list(e2i_default.values())

def test_TNAdapter(cls):
    for dim in [2,3,4]:
        tn = cls()
        logger.debug("Testing dim {}", dim)
        graph = nx.random_regular_graph(3, 10)
        indices = add_random_tn(tn, graph, dim=dim)
        # tn full contraction
        _c_info = tn.optimize()
        ref = tn.contract(_c_info)
        # test slicing
        for index in indices:
            values = []
            for v in range(dim):
                tn2 = tn.slice({index:v})
                _c_info = tn2.optimize()
                logger.debug("Contracting {} with value {}", str(index), v)
                values.append(tn2.contract(_c_info))
            logger.debug("Reference: {}, values: {}", ref, values)
            assert np.allclose(np.sum(values), ref)
        # test free indices and partial contraction
        for _ in range(5):
            # random subset of indices
            __memory_budget = 2**24 # 16MB
            __max_free = int(np.log(__memory_budget) / np.log(dim))
            n_free = np.random.randint(1, min(len(indices), __max_free))
            n_contract = len(indices) - n_free
            contract = np.random.choice(indices, n_contract, replace=False)
            logger.debug("Free indices: {}, contract ({}): {}", n_free, n_contract, contract)
            tn2 = tn.slice({}) # copy
            # sometimes it's better to specify non-contracting indices
            _c_info = tn2.optimize(index_list=contract)
            res = tn2.contract(_c_info)
            logger.debug("Reference: {}, result shape: {}", ref, res.shape)
            assert np.allclose(res.sum(), ref)


        logger.debug("Testing dim {} finished!\n===", dim)

# -- QTensor tensor adapter

class QTensorContractionInfo(ContractionInfo):
    def __init__(self, peo, width):
        self.peo = peo
        self.width = width
    
    def __repr__(self):
        return f"QTensorContractionInfo({self.peo}, {self.width})"

class QTensorTNAdapter(TNAdapter):
    def __init__(self, buckets, data_dict, bra_vars, ket_vars, *args, **kwargs):
        self._indices_to_vars = {}
        self.qtn = qtensor.optimisation.QtreeTensorNet(
            buckets, data_dict, bra_vars, ket_vars
        )
    
    @property
    def _all_indices(self):
        return set(self._indices_to_vars.keys())

    def optimize(self, out_indices: Iterable = [],
                 opt:qtensor.optimisation.Optimizer=qtensor.optimisation.GreedyOptimizer()) -> QTensorContractionInfo:
        logger.trace("Optimizing buckets: {}", self.qtn.buckets)
        free_indices = out_indices
        free_vars = [self._indices_to_vars[i] for i in free_indices]
        self.qtn.free_vars = free_vars
        logger.debug("Free vars: {}", free_vars)
        peo, tn = opt.optimize(self.qtn)
        logger.debug("Contraction path: {}", peo)
        return QTensorContractionInfo(peo, opt.treewidth)
    
    def slice(self, slice_dict: dict):
        slice_dict = {self._indices_to_vars[k]:v for k,v in slice_dict.items()}
        logger.trace("Buckets before slicing: {}", self.qtn.buckets)
        sliced_buckets = self.qtn.backend.get_sliced_buckets(
            self.qtn.buckets, self.qtn.data_dict, slice_dict
        )
        new_tn = QTensorTNAdapter()
        logger.trace("slice dict {}, buckets: {}", slice_dict, sliced_buckets)
        new_tn.qtn.buckets = sliced_buckets
        new_tn.qtn.data_dict = self.qtn.data_dict
        new_tn.qtn.bra_vars = self.qtn.bra_vars
        new_tn.qtn.ket_vars = self.qtn.ket_vars
        new_tn.qtn.backend = self.qtn.backend
        new_tn.qtn.free_vars = self.qtn.free_vars
        new_tn._indices_to_vars = self._indices_to_vars
        return new_tn

    def contract(self, contraction_info: QTensorContractionInfo):
        import qtree
        peo = contraction_info.peo
        if len(self.qtn.buckets) != len(peo):
            _buckets = [[] for _ in peo]
            _buckets[0] = sum(self.qtn.buckets, [])
            self.qtn.buckets = _buckets
        # -- fixer peo
        i_to_var = {int(v): v for v in self._indices_to_vars.values()}
        peo = [i_to_var[int(pv)] for pv in peo]
        #--
        perm_buckets, perm_dict = qtree.optimizer.reorder_buckets(
            self.qtn.buckets, peo
        )
        
        logger.trace("Permuted buckets: {}", perm_buckets)
        sliced_buckets = self.qtn.backend.get_sliced_buckets(
            perm_buckets, self.qtn.data_dict, {}
        )
        be = qtensor.contraction_algos.bucket_elimination
        logger.trace("Sliced buckets: {}", sliced_buckets)
        
        result = be(sliced_buckets, self.qtn.backend,
            n_var_nosum=len(self.qtn.free_vars)
           )
        if isinstance(result, qtree.optimizer.Tensor):
            return result.data
        else:
            return result
    
    def add_tensor(self, data, indices):
        import qtree
        shape = data.shape
        indices_v = {i: qtree.optimizer.Var(i, size=s) for i,s in zip(indices, shape)}
        self._indices_to_vars.update(indices_v)
        indices_v = list(indices_v.values())
        tensor = qtree.optimizer.Tensor(name='t', indices=indices_v, data_key=id(data))
        self.qtn.data_dict[id(data)] = data
        self.qtn.buckets.append([tensor])
        logger.trace("Added tensor {}.", tensor)

    @classmethod
    def from_qtree_gates(cls, qc, init_state=None, **kwargs):
        all_gates = qc
        n_qubits = len(set(sum([g.qubits for g in all_gates], tuple())))
        qtree_circuit = [[g] for g in qc]
        if init_state is None:
            buckets, data_dict, bra_vars, ket_vars = qtree.optimizer.circ2buckets(
                n_qubits, qtree_circuit)
        else:
            buckets, data_dict, bra_vars, ket_vars = circ2buckets_init(
                n_qubits, qtree_circuit, init_vector=init_state)

        tn = cls(buckets, data_dict, bra_vars, ket_vars, **kwargs)
        return tn
