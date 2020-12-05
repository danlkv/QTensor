from qtensor.tools.lazy_import import MPI
import time
from functools import wraps, partial
import numpy as np

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
        server_recvs = [np.ones(1, dtype='i')*-1 for x in range(1, size)]
    client_ack = np.array([0], dtype='i')

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
                        comm.Irecv([server_recvs[other-1], MPI.INT], source=other, tag=1)
                        #comm.irecv(source=other, tag=2)
                    )
                for i in range(len(status_requests)):
                    req = status_requests.pop(0)
                    x = req.Test()
                    #print(i, np.array(server_recvs).flatten())
                    if x:
                        pbar.update(1)
                    else:
                        status_requests.append(req)
                 #  if server_recvs[i][0] is not -1:
                 #      pbar.update(1)
                 #      server_recvs[i][0] = -1
                 #  else:
                 #      pass
                # --
            else:
                if len(status_requests):
                    req = status_requests.pop()
                    #req.wait()
                client_ack[0] = n_call
                req = comm.Isend([client_ack, MPI.INT], dest=0, tag=1)
                req.Wait()
                #req = comm.isend(n_call, dest=0, tag=1)
                status_requests.append(req)
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
