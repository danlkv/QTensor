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

def get_circuit(N, degree, p):
    gamma, beta =  get_gb(p)
    comp = DefaultQAOAComposer(nx.random_regular_graph(degree, N), gamma=gamma, beta=beta)
    comp.ansatz_state()
    return comp.circuit

def tensor_size(t):
    return len(t.indices)
def bucket_size(b):
    if len(b) == 0:
        return 0
    return max(map(tensor_size, b))

# -- Contraction methods

def contract_with_compression(circ, bucket_stop=5, threshold=0.001):
    sim = QAOAQtreeSimulator(DefaultQAOAComposer)
    sim.prepare_buckets(circ)
    buckets = sim.buckets
    # -- Do some contraction, but stop early leaving `bucket_stop` steps
    backend = NumpyBackend()
    qtree.optimizer.bucket_elimination(
        buckets, backend.process_bucket,
        n_var_nosum = bucket_stop,
    )
    # The buckets list now changed and contains large tensors
    # -- Compress the tensors, while tracking the size
    bucket_sizes = [] # list of bucket sizes
    for bucket in buckets:
        for tensor in bucket:
            tensor.data[np.abs(tensor.data) < threshold] = 0
        bucket_sizes.append(bucket_size(bucket)) # size of bucket
    print("Maximum bucket size at compression point:", max(bucket_sizes))

    # -- Finish the contraction
    res = qtree.optimizer.bucket_elimination(
        buckets, backend.process_bucket,
    )
    return res.data

def contract_exact(circ):
    sim = QAOAQtreeSimulator(DefaultQAOAComposer)
    res = sim.simulate(circ)
    print("Contraction width:", sim.optimizer.treewidth)
    return res

# -- 


def main():
    # get buckets for QAOA circuit and ansatz_state simulation
    circ = get_circuit(N=26, degree=3, p=3)
    res_exact = contract_exact(circ)
    # Decreasing the bucket_stop will belate the compression application 
    # thus reducing the error from the compression,
    # but will increase the size of the compressed tensors. 
    # It's important to check the width vs contraction step graph so that
    # not to apply the compression after the large tensors are already contracted.
    bucket_stop = 22
    print('res_exact', res_exact)
    # Note: thresholds should depend on N as the absolute tensor entries will shrink
    # It would be reasonable to use thresholds normalized by 1/2^N
    compression_thresholds = [10**x for x in [-1, -1.5, -2, -2.5, -3, -3.5, -4]]
    for compression_threshold in compression_thresholds:
        res_compressed = contract_with_compression(
            circ, threshold=compression_threshold, bucket_stop=bucket_stop)
        print('threshold', compression_threshold)
        print('res_compressed', res_compressed)
        # Print median error relative to exact result
        abs_error = np.abs(res_exact - res_compressed)
        rel_error = abs_error / np.abs(res_exact)
        print('abs_error', abs_error)
        print('rel_error', rel_error)
        # Note that the result may be a tensor as well (for amplitude batch contraction)
        print('median_rel_error', np.median(rel_error))
        print('max_rel_error', np.max(rel_error))
        print('min_rel_error', np.min(rel_error))
        print()

if __name__=="__main__":
    main()
