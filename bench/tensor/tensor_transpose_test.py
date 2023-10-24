import numpy as np
import time
import sys


def main():
    repeats = 20
    indices_a = 'cegilnoqsuvxzBD'
    indices_b = 'cegilnqsuwyACEF'
    # union of indices_a and indices_b
    indices_result = ''.join(sorted(set(indices_a + indices_b)))

    if len(sys.argv)>1 and sys.argv[1] == "-R":
        print("Decreasing sort. ")
        indices_a = indices_a[::-1]
        indices_b = indices_b[::-1]
        indices_result = indices_result[::-1]
    else:
        print("Increasing sort. ")

    data_a = np.random.rand(*([2] * len(indices_a)))
    data_b = np.random.rand(*([2] * len(indices_a)))
    times = []
    for i in range(repeats):
        start = time.time()
        _ = np.einsum(f"{indices_a},{indices_b}->{indices_result}", data_a, data_b)
        times.append(time.time() - start)

    print(f"Average time: {np.mean(times)}")

if __name__ == '__main__':
    main()
