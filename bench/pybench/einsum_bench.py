import time

import numpy as np

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
    elapsed = benchmark(num_iter, num_batch, np.einsum, "ji,ik", a, b)
    print(n, ", ", nops / elapsed)

if __name__ == "__main__":
    # for i in range(4102, 4090 - 1, -1):
    #     run(i, 20, 1)
    for i in range(500, 1001, 50):
        run(i, 5, 1)
