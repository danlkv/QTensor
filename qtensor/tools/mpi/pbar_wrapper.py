from qtensor.tools.lazy_import import MPI
import time
from functools import wraps, partial

from tqdm.auto import tqdm


def pbar_wrapper(f=None, total=None):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    status_requests = []
    if rank==0:
        pbar = tqdm(total=total)
        # There is usually a delay between calling the wrapper
        # and actually starting work. 
        # tqdm prints stuff when initializing pbar, so
        # any prints after initializing will be on the same line, 
        # which is ugly
        print()

    def wrapper(f):
        n_call = 0
        @wraps(f)
        def wrapee(*args, **kwargs):
            r_ = f(*args, **kwargs)
            start = time.time()
            nonlocal n_call
            n_call += 1
            if rank==0:
                pbar.update(1)
                # -- Non-waiting receival from `size` peers
                # Receive acknowledgements of completing
                for other in range(1, size):
                    status_requests.append(
                        comm.irecv(source=other, tag=1)
                    )
                for req in status_requests:
                    _, data = req.test()
                    if data is not None:
                        pbar.update(1)
                    else:
                        pass
                # --
            else:
                req = comm.isend(n_call, dest=0, tag=1)
            end = time.time()
            wrapee._comm_overhead += end-start

            return r_

        wrapee._comm_overhead = 0
        def gather_comm_overhead(wp):
            overhead = wp._comm_overhead
            overhead = comm.gather(overhead, root=0)
            if rank==0:
                return sum(overhead)

        wrapee.gather_comm_overhead = partial(gather_comm_overhead, wrapee)

        return wrapee
    if f is None:
       return wrapper
    else:
       return wrapper(f)
