import numpy as np
import math
import string
from dataclasses import dataclass
from typing import TypeVar, Generic, Iterable, Tuple

class Array(np.ndarray):
    shape: tuple

D = TypeVar('D') # tensor data type (numpy, torch, etc.)

CHARS = string.ascii_lowercase + string.ascii_uppercase

N = TypeVar('N', bound=np.ndarray)

@dataclass
class Port:
    tensor_ref: int
    ix: int

@dataclass
class ContractionInfo:
    result_indices: Iterable[int]

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

        for idx, slice_val in slice_dict.items():
            # make sure idx is valid
            if idx >= len(tn._edges):
                continue
            
            edge_list = list(tn._edges)
            edge = edge_list.pop(idx)
            # now put the updated edges back on the class
            tn._edges = tuple(edge_list)
            # get all tensors indexed by this edge
            tensors_to_slice = set(port.tensor_ref for port in edge)
            # store slice index and value for each tensor
            local_slices_dict = {}
            for current_tensor_ref in tensors_to_slice:
                slice_dict = {} # TODO: make sure this handles the case with multiple ports pointing to the same tensor
                # get all ports for the current tensor
                current_tensor_ref_ports = [port for port in edge if port.tensor_ref == current_tensor_ref]
                for current_port in current_tensor_ref_ports:
                    slice_dict[current_port.ix] = slice_val
                # store the slice params for this tensor in the local dict
                local_slices_dict[current_tensor_ref] = slice_dict

            # now use the local slice dict to slice for each tensor
            for current_tensor_ref, slice_dict in local_slices_dict.items():
                slice_bounds = []
                current_tensor = tn._tensors[current_tensor_ref]
                for idx in range(current_tensor.ndim):
                    try:
                        slice_bounds.append(slice_dict[idx])
                    except KeyError:
                        slice_bounds.append(slice(None))
                tn._tensors[current_tensor_ref] = tn._tensors[current_tensor_ref][tuple(slice_bounds)]

        return tn

    def copy(self):
        new = TensorNetwork()
        new._tensors = self._tensors[:]
        new._edges = self._edges[:]
        new.shape = self.shape[:]
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
    

    def _get_random_indices_to_contract(self, count=2):
        import random
        tn_copy = self.copy()
        indices_to_contract = []
        counter = 0
        edges_with_indices = [idx for idx, port in enumerate(list(tn_copy._edges))]

        while counter < count and len(edges_with_indices) > 0:
            random_element = random.choice(edges_with_indices)
            edges_with_indices.remove(random_element)
            indices_to_contract.append(random_element)
            counter += 1
        
        return sorted(indices_to_contract)

    # contract to produce a new tensor
    def contract(self, contraction_info: ContractionInfo) -> np.ndarray:
        einsum_expr = self._get_einsum_expr(contraction_info)
        print(einsum_expr)
        print([t.shape for t in self._tensors])
        print(self._edges)
        print(len(self._tensors))
        import pdb; pdb.set_trace()
        try:
            return np.einsum(einsum_expr, *self._tensors)
        except:
            import pdb; pdb.set_trace()

    # for reference, see qtensor/contraction_backends/numpy.py -> get_einsum_expr
    def _get_einsum_expr(self, contraction_info: ContractionInfo) -> str:
        # mapping from tensor index to a tuple of edges that preserves ordering
        # st can lookup tix -> tuple(idx of edges) # this iterable needs to be sorted by of port.ix
        t_ref_to_edges = {}
        # TODO: can do this in a single loop by looping over edges and looking up
        for t_idx in range(0, len(self._tensors)):
            connected_edges = []
            for edge_index, edge in enumerate(self._edges):
                for port in edge:
                    if port.tensor_ref == t_idx:
                        connected_edges.append((edge_index, port.ix))
            # now sort by port ix
            connected_edges_sorted = sorted(connected_edges, key=lambda x: x[1])
            # extract the ix of the global edge
            edge_indices_sorted = [edge_index for edge_index, port_ix in connected_edges_sorted]
            t_ref_to_edges[t_idx] = edge_indices_sorted

        # i:0, j:1, k:2, l:3 -> int is index is self._edges
        # s[0] = (012), s[1]=(103) where index to s is the index in self._tensors
        # edge 0 is (Port(t_ref=0, ix=0), Port(t_ref=1, ix=1)) # i
        # edge 1 is (Port(t_ref=0, ix=1), Port(t_ref=1, ix=0)) # j
        # edge 2 is (Port(t_ref=0, ix=2)) #k 
        # edge 3 is (Port(t_ref=1, ix=2)) #l 

        # TODO: don't need this dict, use chars instead
        edge_to_char = {i: CHARS[i] for i in range(0, len(self._edges))}
        # np.einsum('ijk,jil->jkl', a, b)
        # expr = ','.join(''.join(index_to_char[port.ix] for edge in self._edges for port in edge) for t in self._tensors) + '->' + \
        #    ''.join(index_to_char[ix] for ix in contraction_info.result_indices)

        substrs_to_join = []
        for t_idx, t in enumerate(self._tensors):
            substr = ''
            for edge_idx in t_ref_to_edges[t_idx]:
                substr += edge_to_char[edge_idx]
            substrs_to_join.append(substr)
        
        for ix in contraction_info.result_indices:
            if ix not in edge_to_char:
                raise ValueError("result expects invalid indices")

        expr = ','.join(substrs_to_join) + '->' + ''.join(edge_to_char[ix] for ix in contraction_info.result_indices)
        return expr

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
        # commented out to debug einsum err
        # TODO: need to fix this einsum issue
        # for i in range(len(out.shape)):
        #     eix = partition_fn()
        #     new_port = Port(tensor_ref=-1, ix=i)
        #     partition_dict[eix] = partition_dict.get(eix, [])
        #     partition_dict[eix].append(new_port)

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
    dim = 3
    tn = TensorNetwork.new_random_cpu(2, dim, 4)
    slice_dict = {0: slice(0, 2), 1: slice(1, 3)}
    sliced_tn = tn.slice(slice_dict) # TODO: go through debugger here to make sure that certain edges of the same port aren't being skipped
    # TODO: st i can run contract on a sliced tn without it breaking

    # Where did I leave off?
    # Having trouble verifying tests, perhaps logic is incorrect but it makes sense to me

    # can also do "contract all except..." by knowing indices of edges in tn
    # generate random indices to contract

    random_indices_to_contract = tn._get_random_indices_to_contract(2)
    
    contraction_info = ContractionInfo(tuple(random_indices_to_contract))
    import pdb; pdb.set_trace()
    contracted_tensor = tn.contract(contraction_info)
    print("success")
    import pdb; pdb.set_trace()

"""
dae,dca->be
[(4, 4, 4), (4, 4, 4)]
((Port(tensor_ref=0, ix=1), Port(tensor_ref=1, ix=2)), (Port(tensor_ref=-1, ix=2), Port(tensor_ref=-1, ix=3)), (Port(tensor_ref=1, ix=1),), (Port(tensor_ref=0, ix=0), Port(tensor_ref=1, ix=0)), (Port(tensor_ref=0, ix=2), Port(tensor_ref=-1, ix=1)), (Port(tensor_ref=-1, ix=0),))
2
--Return--
[1] > /app/scratchpad/tn_api/tn.py(160)contract()->None
-> import pdb; pdb.set_trace()
(Pdb++) np.einsum(einsum_expr, *self._tensors)
*** ValueError: einstein sum subscripts string included output subscript 'b' which never appeared in an input
"""