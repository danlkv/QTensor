from mpi4py import MPI
import numpy as np
import time
import sys

from mpi_log_wrapper import decorator as log_decorator

def mpi_map(f, arr, override_size=None):
    """ Map function over array in parallel using MPI. """
    comm = MPI.COMM_WORLD
    size = override_size or comm.Get_size()
    rank = comm.Get_rank()
    if rank==0:
        inputs = [list(arr[x::size]) for x in range(size)]
        if size>len(arr):
            print(f'W:: there are more workers than jobs, {size}>{len(arr)}')
    else:
        inputs = None
    p0 = time.time()
    inputs = comm.scatter(inputs, root=0)
    start = time.time()
    result = list(map(f, inputs))
    end = time.time()
    result = comm.gather(result, root=0)
    p2 = time.time()
    work_time = end-start
    comm_time = p2-end+start-p0
    work_time = comm.gather(work_time, root=0)
    comm_time = comm.gather(comm_time, root=0)
    if rank == 0:
        f._work_time = sum(work_time)
        f._work_std = np.std(work_time)
        f._comm_time = sum(comm_time)
    else:
        f._work_time = None
        f._work_std = None
        f._comm_time = None

    if rank==0:
        return sum(result, [])

@log_decorator(total=100)
def work(N, l=100_000):
    x = np.arange(l)*N
    return sum(np.sin(x)**2 + np.cos(x)**2)/l

x = range(100)
start = time.time()
res = mpi_map(work, x)
end = time.time()
pbar_overhead = work.gather_comm_overhead()
if res:
    print('Result:', sum(res))
    assert(len(res)==len(x)), "Lengths do not match"
    print("progressbar overhead", pbar_overhead)
    print("Comm overhead", work._comm_time)
    print("Work time", work._work_time)
    print("Work time stddev", work._work_std)
    wall_time = end - start
    print("Wall time", wall_time)
    print("Work/Wall time", work._work_time/wall_time)
