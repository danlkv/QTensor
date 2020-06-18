import numpy as np
import time


print('Numpy config:')
np.show_config()


powers = range(3, 10)
nloops = 100
batch = 1000

p_durations = []

for p in powers:
    N = 2**p
    a, b, c = np.random.randn(3,N,N)

    loop_durations = []
    for i in range(nloops):

        start = time.time()
        for j in range(batch):
            np.matmul(a,b,c)
        end = time.time()

        duration_batch = end - start
        duration = duration_batch / batch
        loop_durations.append(duration)
        if duration > 1:
            duration /= 2

    mean = np.mean(loop_durations)
    print(', '.join([str(N), str(mean)]))

    p_durations.append(mean)




