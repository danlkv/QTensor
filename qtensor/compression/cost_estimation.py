from dataclasses import dataclass
from functools import reduce
import numpy as np
from qtensor.optimisation import QtreeTensorNet
from typing import Iterable, Hashable, Dict

Edge = Iterable[Hashable]
Hypergraph = Dict[Hashable, Edge]
# # self = hypergraph
# verts = set(sum(self.values(), []))
# num_edges = len(self)

@dataclass
class Cost:
    use_log = True
    flops: int
    memory: int
    width: int
    compressions: int
    decompressions: int

    def time(self, flops_second, compression_throughput, decompression_throughput, memory_limit):
        """Returns the time in seconds to perform the contraction"""
        return (
            self.flops / flops_second
            + self.compressions *2**memory_limit/ compression_throughput
            + self.decompressions *2**memory_limit/ decompression_throughput
        )

    def __add__(self, other):
        return Cost(
            self.flops + other.flops,
            max(self.memory, other.memory),
            max(self.width, other.width),
            self.compressions + other.compressions,
            self.decompressions + other.decompressions,
        )

    def format_number(self, n):
        if self.use_log:
            return f"{np.log2(n):.2f}"
        else:
            return f"{n}"

    def __str__(self):
        flops_str = self.format_number(self.flops)
        mems_str = self.format_number(self.memory)
        return f"Cost(FLOPs={flops_str}, Memory={mems_str}, width={self.width}, compressions={self.compressions}, decompressions={self.decompressions})"

def dual_hg(hg: Hypergraph) -> Hypergraph:
    dual = {}
    for iedge, edge in hg.items():
        for vert in edge:
            if dual.get(vert) is None:
                dual[vert] = []
            dual[vert].append(iedge)
    return dual

def remove_vertices_tensors(TN, dual_TN, vertices=[], tensors=[]):
    for t in tensors:
        # -- remove tensor
        for v in dual_TN[t]:
            TN[v].remove(t)
        del dual_TN[t]

    for vertex in vertices:
        # remove vertex
        for t in TN[vertex]:
            dual_TN[t].remove(vertex)
        del TN[vertex]

def tn2tn(tn: QtreeTensorNet, peo=None): 
    ignored_vars = tn.bra_vars + tn.ket_vars
    # Vertices --> indices
    # Edges --> tensors
    dual_tn = { str(hex(id(t))):[x for x in t.indices if x not in ignored_vars]
               for t in tn.tensors }

    # Vertices --> tensors
    # Edges --> indices
    TN = dual_hg(dual_tn)
    return TN

def tensor_memory(indices, mem_limit, compression_ratio):
    if len(indices) > mem_limit:
        return 2**len(indices)/compression_ratio
    else:
        return 2**len(indices)
def pairwise_cost(indices, comp_ixs, contracted_ixs=[],
                  mem_limit=np.inf,
                  compression_ratio=30,
                 ):
    """
    Computes the cost of contracting a pair of tensors, assuming last
    `contracted_ixs_count` indices are contrated
    """
    contracted_ixs_count = len(contracted_ixs)
    all_indices = set().union(*indices) 
    next_indices = list(all_indices)
    next_indices.sort(key=int, reverse=True)
    for i in contracted_ixs:
        next_indices.remove(i)

    if len(next_indices) > mem_limit or any(comp_ixs):
        next_comp_ixs= next_indices[:-mem_limit]
        rm_comp = set().union(*comp_ixs) - set(next_comp_ixs)
        decompressions = 2**(len(rm_comp) + len(next_comp_ixs))
        compressions = 2**len(next_comp_ixs)
    else:
        next_comp_ixs = []
        decompressions = 0
        compressions = 0
    mem = 0
    for ilist in [next_indices]+indices:
        mem += tensor_memory(ilist, mem_limit, compression_ratio)

    return (
        next_indices,
        next_comp_ixs,
        Cost(
            memory = mem,
            flops = 2**len(all_indices),
            width = len(next_indices),
            compressions = compressions,
            decompressions = decompressions,
        )
    )


def bucket_contract_cost(indices, comp_ixs, contracted_indices, **kwargs):
    """
    Computes the cost of contracting a bucket of tensors

    Args:
        indices: indices of tensors in the bucket
        comp_ixs: indices that are compressed
        contracted_indices: indices that are contracted
        **kwargs: passed to pairwise_cost
    """
    ixs, compixs = indices[0], comp_ixs[0]
    costs = []
    for i in range(1, len(indices)-1):
        ixs, compixs, cost = pairwise_cost(
            [ixs, indices[i]],
            [compixs, comp_ixs[i]],
            **kwargs
        )
        costs.append(cost)
    # -- contract last two tensors
    new_ixs, new_comp_ixs, cost = pairwise_cost(
        [ixs, indices[-1]],
        [compixs, comp_ixs[-1]],
        contracted_ixs=contracted_indices,
        **kwargs,
    )
    costs.append(cost)
    new_ixs = set().union(*indices) - set(contracted_indices)
    sum_cost = sum(costs[1:], costs[0])
    sum_cost.width = len(new_ixs)
    ## Naive Flops calculation
    # sum_cost.flops = 2**len(set().union(*indices))*(len(indices)+1)
    return new_ixs, new_comp_ixs, sum_cost

def contract_with_cost(TN, comp_ixs, dual_TN, vertex,
                       mem_limit=np.inf,
                       compression_ratio=100):
    """
    Contracts vertex from TN
    TN is a mapping from indices to [tensor]
    """
    tensors = TN[vertex]
    # contract
    tensors.sort(key=lambda t: len(dual_TN[t]))
    indices = [dual_TN[t] for t in tensors]
    comp_indices = [comp_ixs.get(t, []) for t in tensors]
    result_ixs, compressed, cost = bucket_contract_cost(indices, comp_indices, [vertex],
                                                        mem_limit=mem_limit,
                                                        compression_ratio=compression_ratio
                                                       )
    # calculate current memory
    for t_id, indices in dual_TN.items():
        if t_id in tensors:
            # these tensors are accounted in bucket_contract_cost
            continue
        cost.memory += tensor_memory(indices, mem_limit, compression_ratio)

    # This can be random but should be unique
    tensor_id = str(hex(id(vertex)))
    comp_ixs[tensor_id] = compressed
    remove_vertices_tensors(TN, dual_TN, [vertex], tensors)
    # -- add result
    for ix in result_ixs:
        if TN.get(ix) is None:
            TN[ix] = []
        TN[ix].append(tensor_id)
    dual_TN[tensor_id] = list(result_ixs)
    # --
    return cost


def compressed_contraction_cost(tn, peo, mem_limit=np.inf):
    """
    Compute the cost of a contraction with compression.
    """
    TN = tn2tn(tn)
    ignored_vars = tn.bra_vars + tn.ket_vars
    peo = [x for x in peo if x not in ignored_vars]
    costs = []
    dual_TN = dual_hg(TN)
    comp_ixs = {}
    for i in peo:
        cost = contract_with_cost(TN, comp_ixs, dual_TN, i, mem_limit)
        costs.append(cost)
    return costs
