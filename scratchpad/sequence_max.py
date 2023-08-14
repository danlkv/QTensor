import numpy as np
import networkx as nx
from loguru import logger
import qtree, qtensor
import sys
from typing import Iterable, Union
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

import qtensor

class QTensorContractionInfo(ContractionInfo):
    def __init__(self, peo, width):
        self.peo = peo
        self.width = width
    
    def __repr__(self):
        return f"QTensorContractionInfo({self.peo}, {self.width})"

class QTensorTNAdapter(TNAdapter):
    def __init__(self, *args, **kwargs):
        buckets = []
        data_dict = {}
        bra_vars = []
        ket_vars = []
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

def test_QTensorTNAdapter():
    test_TNAdapter(QTensorTNAdapter)

# -- Quimb Tensor adapter

import quimb
import cotengra
import quimb.tensor
from opt_einsum.contract import PathInfo as QuimbPathInfo

class QuimbContractionInfo(ContractionInfo):
    width: int

    def __init__(self, info: QuimbPathInfo):
        self._info = info
        self.width = np.log2(float(info.largest_intermediate))
    
    def __repr__(self):
        return str(self._info)

class QuimbTNAdapter(TNAdapter):
    def __init__(self, tn=None, *args, **kwargs):
        self._indices_to_vars = {}
        if tn is None:
            self.tn = quimb.tensor.TensorNetwork()
        else:
            self.tn = tn
    
    @property
    def _all_indices(self):
        return set(self._indices_to_vars.keys())

    def optimize(self, out_indices: Iterable = [],
                 opt: cotengra.HyperOptimizer = cotengra.HyperOptimizer()
                 ) -> QuimbContractionInfo:
        logger.trace("Optimizing tn: {}", self.tn)
        free_indices = out_indices
        free_vars = [self._indices_to_vars[i] for i in free_indices]
        logger.debug("Free vars: {}", free_vars)
        contract_info = self.tn.contract(
            get='path-info',
            output_inds=free_vars,
            optimize=opt)
        opt_info = QuimbContractionInfo(contract_info)
        logger.debug("Contraction path width: {}", opt_info.width)
        return opt_info
    
    def slice(self, slice_dict: dict):
        slice_dict = {self._indices_to_vars[k]:v for k,v in slice_dict.items()}
        new_tn = self.tn.isel(slice_dict)
        new_tn_adapter = QuimbTNAdapter(tn=new_tn)
        new_tn_adapter._indices_to_vars = self._indices_to_vars
        return new_tn_adapter

    def contract(self, contraction_info: QuimbContractionInfo):
        return self.tn.contract(optimize=contraction_info._info.path)
    
    def add_tensor(self, data, indices):
        indices_v = {i: f'i{i}' for i in indices}
        self._indices_to_vars.update(indices_v)
        indices_v = list(indices_v.values())
        tensor = quimb.tensor.Tensor(data=data, inds=indices_v)
        self.tn.add_tensor(tensor)
        logger.trace("Added tensor {}.", tensor)

def test_QuimbTensorTNAdapter():
    test_TNAdapter(QuimbTNAdapter)

# --

class Bs:
    """Bitstring class"""
    def __init__(self, bits: 'list[int]', prob=None, dim=2):
        self._bits = list(bits)
        self._s = ''.join(str(b) for b in self._bits)
        self._prob = prob
        self._dim = dim
    
    def __iter__(self):
        for i in self._bits:
            yield int(i)

    def __repr__(self):
        return f'<{self._s}>'
    
    def __len__(self):
        return len(self._bits)
    
    @classmethod
    def str(cls, s: str, **kwargs):
        return cls([int(_s) for _s in s], **kwargs)
    
    @classmethod
    def int(cls, i: int, width, **kwargs):
        dim = kwargs.get('dim', 2)
        return cls(list(int(_i) for _i in np.unravel_index(i, [dim]*width)), **kwargs)
    
    def __add__(self, other: 'Bs'):
        assert self._dim == other._dim
        if self._prob is not None and other._prob is not None:
            return Bs(self._bits + other._bits, self._prob * other._prob, dim=self._dim)
        return Bs(self._bits + other._bits, dim=self._dim)

    def __iadd__(self, other: 'Bs'):
        assert self._dim == other._dim
        if self._prob is not None and other._prob is not None:
            self._prob *= other._prob
        self._bits += other._bits

    def __eq__(self, other: 'Bs'):
        return self.to_int() == other.to_int()
    
    def __hash__(self):
        if self._prob is not None:
            return hash((self._s, self._prob, self._dim))
        return int(self.to_int())
    
    def to_int(self):
        return np.ravel_multi_index(self._bits, [self._dim]*len(self._bits))

def contract_tn(tn, slice_dict, free_indices):
    tn_ = tn.slice(slice_dict)
    c_info = tn_.optimize(free_indices)
    return tn_.contract(c_info)

def sequence_sample(tn: TNAdapter, _all_indices, indices, batch_size=10, batch_fix_sequence=None, dim=2):
    """
        Args:
        tn: tensor network
        indices: list of indices to contract
        api: api calls for the TN
    """
    K = int(np.ceil(len(indices) / batch_size))
    if batch_fix_sequence is None:
        batch_fix_sequence = [1]*K
    
    slice_dict = {}
    cache = {}
    samples = [Bs.str('', prob=1., dim=dim)]
    z_0 = None
    for i in range(K):
        for j in range(len(samples)):
            bs = samples.pop(0)
            res = None
            if len(bs)>0:
                res = cache.get(bs.to_int())
            if res is None:
                free_batch_ix = indices[i*batch_size:(i+1)*batch_size]
                _fix_indices = indices[: len(bs)]
                update = dict(zip(_fix_indices, list(bs)))
                slice_dict.update(dict(zip(_fix_indices, list(bs))))
                res = contract_tn(tn, slice_dict, free_batch_ix)
                res = res.real**2
                if len(bs)>0:
                    cache[bs.to_int()] = res
                
            # result should be shaped accourdingly
            if z_0 is None:
                z_0 = res.sum()
            prob_prev = bs._prob
            z_n = prob_prev * z_0
            z_n = res.sum()
            logger.debug('bs {}, Sum res {}, prev_Z {}, prob_prev {}',
                         bs, res.sum(), prob_prev*z_0, prob_prev
                        )
            pdist = res.flatten() / z_n
            #logger.debug(f'Prob distribution orig: {pdist.round(4)}')
            # normalize pdist
            # pdist_z = np.exp(20*pdist/np.mean(pdist))
            # pdist_z /= pdist_z.sum()
            # pdist = pdist_z
            #
            logger.debug(f'Prob distribution: {pdist.round(4)}')
            indices_bs = np.arange(len(pdist))
            batch_ix = np.random.choice(indices_bs, batch_fix_sequence[i], p=pdist)
            for ix in batch_ix:
                _new_s = bs + Bs.int(ix, width=len(free_batch_ix), prob=pdist[ix], dim=dim)
                logger.trace(f'New sample: {_new_s}')
                samples.append(_new_s)

    return samples

            

def test_sequence_sample(cls=QTensorTNAdapter):
    tn = cls()
    dim = 2
    graph = nx.random_regular_graph(3, 10)
    indices = add_random_tn(tn, graph, dim=dim)
    K = 5
    indices_sample = [indices[0], indices[4], indices[6]]
    batch_sequence = [20, 40]
    samples = []
    for i in range(20):
        samples_local = sequence_sample(tn, [], indices_sample, batch_size=2, batch_fix_sequence=batch_sequence)
        samples+=(samples_local)
        assert len(samples_local) == np.prod(batch_sequence)
    bins = range(2**len(indices_sample))
    hist_steps = np.histogram([b.to_int() for b in samples], bins=bins)
    samples_ref = sequence_sample(tn, [], indices_sample, batch_size=len(indices_sample), batch_fix_sequence=[len(samples)])
    hist_ref = np.histogram([b.to_int() for b in samples_ref], bins=bins)
    print('Stepped histogram', hist_steps)
    print('reference histogram', hist_ref)

def test_sequence_sample_bitstring_probs(cls=QTensorTNAdapter):
    tn = cls()
    dim = 3
    graph = nx.random_regular_graph(3, 10)
    indices = add_random_tn(tn, graph, dim=dim)
    K = 5
    indices_sample = [indices[0], indices[4], indices[6]]
    batch_sequence = [4, 4]
    samples = []
    bitstrings = {}
    for i in range(2):
        samples_local = sequence_sample(tn, [], indices_sample, batch_size=2, batch_fix_sequence=batch_sequence, dim=dim)
        samples+=(samples_local)
        for s in samples_local:
            bitstrings[s.to_int()] = s._prob
        assert len(samples_local) == np.prod(batch_sequence)
    samples_ref = sequence_sample(tn, [], indices_sample, batch_size=len(indices_sample), batch_fix_sequence=[len(samples)], dim=dim)
    bitstrings_checked = {}
    for s_ref in samples_ref:
        calc_prob = bitstrings.get(s_ref.to_int(), 0)
        if calc_prob == 0:
            print(f'Warning: bitstring {s_ref} not found in samples')
        else:
            assert np.allclose(calc_prob, s_ref._prob), f'bitstring {s_ref} has prob {s_ref._prob} but should have {calc_prob}'
            bitstrings_checked[s_ref.to_int()] = calc_prob
    logger.info(f'Checked {len(bitstrings_checked)} bitstrings: {bitstrings_checked}')
        
if __name__=="__main__":
    #test_QTensorTNAdapter()
    test_QuimbTensorTNAdapter()
    #test_sequence_sample()
    #test_sequence_sample_bitstring_probs()