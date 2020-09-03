import time

import numpy as np
import cython.view cimport array as cvarray

def benchmark(num_iter, num_batch, f, *args):
    acc = 0;
    for idx in range(num_iter):
        start = time.time()
        for jdx in range(num_batch):
            f(*args)
        acc += (time.time() - start) / num_batch
    return acc / num_iter

def run(n, num_iter, num_batch):
    a, b, c = np.random.randn(3,n,n)
    nops = 2 * n ** 3 / 1e9
    elapsed = benchmark(num_iter, num_batch, np.matmul, a, b, c)
    print(n, ", ", nops / elapsed)

if __name__ == "__main__":
    for i in range(4096, 512 - 256, -256):
        run(i, 10, 1)

    for i in range(512, 64 - 32, -32):
        run(i, 50, 100)

    for i in range(64, 16 - 1, -1):
        run(i, 50, 100)

