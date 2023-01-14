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
            self.memory + other.memory,
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

def tn2tn(tn: QtreeTensorNet, peo=None): 
    ignored_vars = tn.bra_vars + tn.ket_vars
    # Vertices --> indices
    # Edges --> tensors
    dual_tn = { str(hex(id(t))):[x for x in t.indices if x not in ignored_vars]
               for t in tn.tensors }
    if peo:
        dual_tn = { str(hex(id(t))):[peo.index(x) for x in t.indices if x not in ignored_vars]
                   for t in tn.tensors }

    # Vertices --> tensors
    # Edges --> indices
    TN = dual_hg(dual_tn)
    return TN

def pairwise_cost(indices, comp_ixs, mem_limit, contracted_ixs_count=0):
    """
    Computes the cost of contracting a pair of tensors, assuming last
    `contracted_ixs_count` indices are contrated
    """
    all_indices = set().union(*indices) 
    next_indices = list(all_indices)
    next_indices.sort(key=int, reverse=True)
    next_indices = next_indices[:-contracted_ixs_count]

    if len(next_indices) > mem_limit or any(comp_ixs):
        next_comp_ixs= next_indices[:-mem_limit]
        rm_comp = set().union(*comp_ixs) - set(next_comp_ixs)
        decompressions = 2**(len(rm_comp)+len(next_comp_ixs))
        compressions = 2**len(next_comp_ixs)
    else:
        next_comp_ixs = []
        decompressions = 0
        compressions = 0
    return (
        next_indices,
        next_comp_ixs,
        Cost(
            memory = 2**len(next_indices),
            flops = 2**len(all_indices),
            width = len(next_indices),
            compressions = compressions,
            decompressions = decompressions,
        )
    )


def bucket_contract_cost(indices, comp_ixs, mem_limit):
    ixs, compixs = indices[0], comp_ixs[0]
    costs = []
    for i in range(1, len(indices)-1):
        ixs, compixs, cost = pairwise_cost(
            [ixs, indices[i]],
            [compixs, comp_ixs[i]],
            mem_limit, contracted_ixs_count=0
        )
        costs.append(cost)
    new_ixs, new_comp_ixs, cost = pairwise_cost(
        [ixs, indices[-1]],
        [compixs, comp_ixs[-1]],
        mem_limit, contracted_ixs_count=1
    )
    costs.append(cost)
    return new_ixs, new_comp_ixs, sum(costs[1:], costs[0])

def contract_with_cost(TN, comp_ixs, dual_TN, vertex, mem_limit=30):
    """
    Contracts vertex from TN
    TN is a mapping from indices to [tensor]
    """
    tensors = TN[vertex]
    # contract
    tensors.sort(key=lambda t: len(dual_TN[t]))
    indices = [dual_TN[t] for t in tensors]
    comp_itensors = [comp_ixs.get(t, []) for t in tensors]
    _, compressed, cost = bucket_contract_cost(indices, comp_itensors, mem_limit)

    result_ixs = set().union(*indices)
    result_ixs.remove(vertex)
    # This can be random but should be unique
    tensor_id = str(hex(id(vertex)))
    comp_ixs[tensor_id] = compressed
    # -- remove tensors
    for t in tensors:
        for v in dual_TN[t]:
            TN[v].remove(t)
        del dual_TN[t]
    # remove vertex
    for t in TN[vertex]:
        dual_TN[t].remove(vertex)
    del TN[vertex]
    # -- add result
    for ix in result_ixs:
        if TN.get(ix) is None:
            TN[ix] = []
        TN[ix].append(tensor_id)
    dual_TN[tensor_id] = list(result_ixs)
    # --
    return cost


def compressed_contraction_cost(tn, peo, mem_limit=None):
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
