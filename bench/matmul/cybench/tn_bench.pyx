import time
import sys

import numpy as np
import tensornetwork as tn

def benchmark(num_iter, num_batch, f, *args):
    acc = 0;
    for idx in range(num_iter):
        start = time.time()
        for jdx in range(num_batch):
            f(*args)
        acc += (time.time() - start) / num_batch
    return acc / num_iter

def foo(a, b):
    node1 = tn.Node(a)
    node2 = tn.Node(b)
    node1[1] ^ node2[0]
    return tn.contract_between(node1, node2)


def run(n, num_iter, num_batch):
    a, b = np.random.randn(2,n,n)
    nops = 2 * n ** 3 / 1e9
    foo(a, b)
    elapsed = benchmark(num_iter, num_batch, foo, a, b)
    print(n, ", ", nops / elapsed)

if __name__ == "__main__":
    tn.set_default_backend(sys.argv[1])
    for i in range(4096, 512 - 256, -256):
        run(i, 10, 1)

    for i in range(512, 64 - 32, -32):
        run(i, 50, 100)

    for i in range(64, 16 - 1, -1):
        run(i, 50, 100)

