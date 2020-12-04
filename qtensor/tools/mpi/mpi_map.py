from qtensor.tools.lazy_import import MPI
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
    w = MPI.COMM_WORLD
    comm = MPI.Comm
    size = override_size or comm.Get_size(w)
    rank = comm.Get_rank(w)
    if rank==0:
        #indices = arr #list(range(len(arr)))
        indices = list(range(len(arr)))
        """
        Instead of sending the data itself, we can just send indices that will be used,
        since each mpi rank has its copy of input data
        """
        input_indices = [list(indices[x::size]) for x in range(size)]
        lens = [len(x) for x in input_indices]
        print(f'I:: There are {size} workers, each will get {np.mean(lens)} tasks on average.', flush=True)
        if size>len(arr):
            print(f'W:: there are more workers than jobs, {size}>{len(arr)}')
    else:
        input_indices = None
    p0 = time.time()
    input_indices = w.scatter(input_indices, root=0)
    """
    Get the input arguments assigned for current mpi rank
    """
    #inputs = input_indices 
    inputs = [arr[i] for i in input_indices]
    start = time.time()
    result = list(map(f, inputs))
    end = time.time()
    result = w.gather(result, root=0)
    p2 = time.time()
    work_time = end-start
    comm_time = p2-end+start-p0
    work_time = w.gather(work_time, root=0)
    comm_time = w.gather(comm_time, root=0)

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
    def work(N, l=50_000):
        l = l+ np.random.randint(l)
        x = np.arange(l)*N[0]
        return sum(np.sin(x)**2 + np.cos(x)**2)/l

    # the large inputs are emulated to check that 
    # data is not sent, only indices
    x = [np.ones(50_000)*i for i in range(100)]
    res = mpi_map(work, x)
    if res:
        print('Result:', sum(res))
        assert(len(res)==len(x)), "Lengths do not match"
        print_stats()

if __name__=='__main__':
    test()
