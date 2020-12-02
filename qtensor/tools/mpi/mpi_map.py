from qtensor.tools.lazy_import import mpi4py
import numpy as np
import sys
import time
from qtensor.tools.mpi import pbar_wrapper

RECENT_TASK = None

def mpi_map(f, arr, override_size=None, pbar=False, total=None):
    if pbar:
        f = pbar_wrapper(total=total)(f)
    return _mpi_map(f, arr, override_size=override_size)

def _mpi_map(f, arr, override_size=None):
    """ Map function over array in parallel using MPI. """
    comm = mpi4py.MPI.COMM_WORLD
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

    f._wall_time = end-start
    f._comm_size = size
    if rank == 0:
        f._work_time = sum(work_time)
        f._work_std = np.std(work_time)
        f._comm_time = sum(comm_time)
        f._wall_time = end-start
    else:
        f._work_time = None
        f._work_std = None
        f._comm_time = None
    try:
        f._pbar_overhead = f.gather_comm_overhead()
    except:
        # We got just a regular function 
        f._pbar_overhead = 0
        pass

    global RECENT_TASK
    RECENT_TASK = f

    if rank==0:
        return sum(result, [])

def print_stats():
    global RECENT_TASK
    if RECENT_TASK is None:
        print('There is no recent task', file=sys.stderr)
        return

    pbar_overhead = RECENT_TASK._pbar_overhead
    print("Processes", RECENT_TASK._comm_size)
    print("progressbar overhead", pbar_overhead)
    print("Comm overhead", RECENT_TASK._comm_time)
    print("Work time", RECENT_TASK._work_time)
    print("Work time stddev", RECENT_TASK._work_std)
    print("Wall time", RECENT_TASK._wall_time)
    print("Work/Wall time", RECENT_TASK._work_time/RECENT_TASK._wall_time)


def test():
    print('Testing mpi map')
    @pbar_wrapper(total=100)
    def work(N, l=100_000):
        x = np.arange(l)*N
        return sum(np.sin(x)**2 + np.cos(x)**2)/l

    x = range(100)
    res = mpi_map(work, x)
    if res:
        print('Result:', sum(res))
        assert(len(res)==len(x)), "Lengths do not match"
        print_stats()

if __name__=='__main__':
    test()
