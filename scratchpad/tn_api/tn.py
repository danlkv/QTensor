import numpy as np
import math
from dataclasses import dataclass
from typing import TypeVar, Generic, Iterable

class Array(np.ndarray):
    shape: tuple

D = TypeVar('D') # tensor data type (numpy, torch, etc.)

class ContractionInfo:
    pass

class TensorNetworkIFC(Generic[D]):
    def __init__(self, *args, **kwargs):
        ...

    def optimize(self, out_indices: Iterable = []) -> ContractionInfo:
        return ContractionInfo()
    
    # slice not inplace
    def slice(self, slice_dict: dict) -> 'TensorNetwork':
        ...

    # contract to produce a new tensor
    def contract(self, contraction_info: ContractionInfo) -> D:
        ...

    # 
    def copy(self):
        ...

    def add(self, other: "TensorNetworkIFC[D] | D"):
        ...


    @classmethod
    def new_random_cpu(cls, dims: Iterable[int])-> 'TensorNetworkIFC[D]':
        ...

    def __eq__(a, b):
        ...


N = TypeVar('N', bound=np.ndarray)

@dataclass
class Port:
    tensor_ref: int
    ix: int

class TensorNetwork(TensorNetworkIFC[np.ndarray]):
    tensors: Iterable[np.ndarray]
    shape: tuple
    edges: tuple

    def __init__(self, *args, **kwargs):
        self._tensors = []
        self._edges = tuple()
        self.shape = tuple()

    # slice not inplace
    def slice(self, slice_dict: dict) -> 'TensorNetwork':
        tn = self.copy()
        sliced_tns = []
        for tensor in tn._tensors:
            slice_bounds = []
            for idx in range(tensor.ndim):
                try:
                    slice_bounds.append(slice_dict[idx])
                except KeyError:
                    slice_bounds.append(slice(None))

            sliced_tns.append(tensor[tuple(slice_bounds)])

        tn._tensors = sliced_tns
        return tn

    def copy(self):
        new = TensorNetwork()
        new._tensors = self._tensors
        new._edges = self._edges
        new.shape = self.shape
        return new

    def add(self, other: "TensorNetwork | np.ndarray"):
        if not isinstance(other, TensorNetwork):
            self._tensors.append(other)
            self.shape = self.shape + other.shape
        else:
            m = len(self._tensors)
            n = len(self.shape)
            # -- other's edges tensors will refer to shifted tensor location
            enew = []
            for e in other._edges:
                e_ = []
                for p in e:
                    if p.tensor_ref == -1:
                        e_.append(Port(tensor_ref=-1, ix=p.ix+n))
                    else:
                        e_.append(Port(tensor_ref=p.tensor_ref+m, ix=p.ix))
                enew.append(tuple(e_))

            self._edges += tuple(enew)
            self._tensors += other._tensors
            self.shape += other.shape

    # contract to produce a new tensor
    def contract(self, contraction_info: ContractionInfo) -> np.ndarray:
        raise NotImplementedError()

    def optimize(self, out_indices: Iterable = []) -> ContractionInfo:
        raise NotImplementedError()
    

    @classmethod
    def new_random_cpu(cls, count, size, dim: int):
        out = cls()
        for i in range(count):
            t: np.ndarray = np.random.random((dim, )*size)
            out.add(t)
        # arbitrary number of output indices
        out_dims = np.random.randint(low=0, high=len(out.shape))
        tensor_dims = len(out.shape)
        out.shape = (dim, )*out_dims
        # -- random connectivity
        # A hypergraph can be generated as a partition into
        # E parts where E is number of edges. The isolated vertices are equivalent
        # to vertices with 1 edge that contains only them.
        # arbitrary max number of edges, must be less than total indices
        edges_cnt = np.random.randint(low=1, high=tensor_dims+out_dims)
        # a partition can be implemented using a random function
        partition_fn = lambda : np.random.randint(low=0, high=edges_cnt)
        partition_dict = {}
        for t_ref, t in enumerate(out._tensors):
            for i in range(t.ndim):
                eix = partition_fn()
                new_port = Port(tensor_ref=t_ref, ix=i)
                partition_dict[eix] = partition_dict.get(eix, [])
                partition_dict[eix].append(new_port)

        # add "self" tensor indices to partition
        for i in range(len(out.shape)):
            eix = partition_fn()
            new_port = Port(tensor_ref=-1, ix=i)
            partition_dict[eix] = partition_dict.get(eix, [])
            partition_dict[eix].append(new_port)

        edges = []
        for i in range(edges_cnt):
            p = partition_dict.get(i)
            if p is not None:
                edges.append(tuple(p))
        out._edges = tuple(edges)
        return out

    def __eq__(self, other):
        if self.shape != other.shape:
            return False
        if self._edges != other._edges:
            return False
        return all((a==b).all() for a, b in zip(self._tensors, other._tensors))

    def __repr__(self):
        return f"TensorNetwork({self.shape})<{self._tensors}, {self._edges}>"



if __name__ == "__main__":
    tn = TensorNetwork.new_random_cpu(2, 3, 4)
    slice_dict = {0: slice(0, 2), 1: slice(1, 3)}
    sliced_tn = tn.slice(slice_dict)
    import pdb; pdb.set_trace()