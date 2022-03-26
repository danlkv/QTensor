import qtensor
import numpy as np
from qtensor import QAOAQtreeSimulator, DefaultQAOAComposer
from qtensor.contraction_backends import NumpyBackend
import qtree
import random
import networkx as nx
SEED = 10
np.random.seed(SEED)
random.seed(SEED)

def get_gb(p):
    return [0.1]*p, [0.2]*p

def get_buckets(N, degree, p):
    sim = QAOAQtreeSimulator(DefaultQAOAComposer)
    gamma, beta =  get_gb(p)
    comp = DefaultQAOAComposer(nx.random_regular_graph(degree, N), gamma=gamma, beta=beta)
    comp.ansatz_state()
    sim.prepare_buckets(comp.circuit)
    buckets = sim.buckets
    # -- Do some contraction to populate buckets
    backend = NumpyBackend()
    qtree.optimizer.bucket_elimination(
        buckets, backend.process_bucket,
        n_var_nosum = 5
    )
    return buckets


def tensor_size(t):
    return len(t.indices)
def bucket_size(b):
    return max(map(tensor_size, b))

class DIMS:
    k = 1000
    M = k*1000
    G = M*1000

    @classmethod
    def get_tw_mem(cls, mem):
        return int(np.log2(mem/16))

def save_tensor(prefix, tensor):
    nix = len(tensor.indices)
    data = tensor.data
    dtype = str(data.dtype)
    filename = f"{prefix}_dims-{nix}_dtype-{dtype}.bin"
    data.tofile(filename)

def main():
    # get buckets for QAOA circuit and ansatz_state simulation
    buckets = get_buckets(N=32, degree=3, p=3)
    i = 2
    bucket_widths = np.array(list(map(bucket_size, buckets)))
    print('Bucket sizes', bucket_widths)
    # sizes in bytes
    requested_sizes = [100, 400, 100*DIMS.M]
    filename_prefix = 'tensor'
    for size in requested_sizes:
        tw = DIMS.get_tw_mem(size)
        print(f'TW needed for {size} bytes', tw)
        _deltas = np.abs(bucket_widths - tw)
        _bucket_ix = np.argmin(_deltas)
        _bucket_select = buckets[_bucket_ix]
        _tensor_sizes = np.array(list(map(tensor_size, _bucket_select)))
        _tensor_ix = np.argmax(_tensor_sizes)
        _tensor_select = _bucket_select[_tensor_ix]
        print('Tensor select ixs', _tensor_select.indices)
        # -- Saving
        save_tensor(filename_prefix, _tensor_select)





if __name__=="__main__":
    main()
