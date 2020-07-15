import qtree
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

from qensor.ProcessingFrameworks import NumpyBackend
from qensor.Simulate import Simulator, QtreeSimulator
from qensor.optimisation.Optimizer import SlicesOptimizer
from qensor.optimisation.TensorNet import QtreeTensorNet
import psutil

from loguru import logger as log

from qensor import utils

def int_slice(value, vars_to_slice):
    dimensions = [var.size for var in vars_to_slice]
    multiindex = qtree.utils.unravel_index(value, dimensions)

    return {var: at for var, at in zip(vars_to_slice, multiindex)}

class FeynmanSimulator(QtreeSimulator):
    optimizer = SlicesOptimizer
    opt_args = {}


    def optimize_buckets(self, fixed_vars: list=None):
        opt_args = {'tw_bias': self.tw_bias}
        opt_args.update(self.opt_args)
        opt = self.optimizer(**opt_args)
        peo, par_vars, self.tn = opt.optimize(self.tn)
        self.parallel_vars = par_vars
        return peo

    def _parallel_unit(self, par_idx):
        slice_dict = self._get_slice_dict(par_state=par_idx)

        sliced_buckets = self.tn.slice(slice_dict)
        result = qtree.optimizer.bucket_elimination(
            sliced_buckets, self.bucket_backend.process_bucket
        , n_var_nosum=len(self.tn.free_vars + self.parallel_vars))
        return result.data.flatten()

    def simulate(self, qc, batch_vars=0, tw_bias=2):
        return self.simulate_batch_adaptive(qc, batch_vars, tw_bias=tw_bias)


    def simulate_batch_adaptive(self, qc, batch_vars=0, tw_bias=2):
        self.tw_bias = tw_bias
        self._new_circuit(qc)
        self._create_buckets()
        # Collect free qubit variables
        free_final_qubits = list(range(batch_vars))
        log.info("Free qubit variables: {}", free_final_qubits)
        self._set_free_qubits(free_final_qubits)
        self._optimize_buckets()

        self._reorder_buckets()

        n_processes = 2
        with Pool(n_processes) as p:
            total_paths = 2**len(self.parallel_vars)
            log.info('Starting to simulate {} paths using {} processes', total_paths, n_processes)
            args = range(total_paths)
            piter = p.imap(self._parallel_unit, args)
            r = list(tqdm(piter, total=total_paths))
            #r = list(piter)
        result = sum(r)
        return result

    def _reorder_buckets(self):
        perm_dict = super()._reorder_buckets()
        self.parallel_vars = sorted([perm_dict[idx] for idx in self.parallel_vars], key=str)

    def _get_slice_dict(self, initial_state=0, target_state=0, par_state=0):
        slice_dict = super()._get_slice_dict(initial_state, target_state)
        slice_dict.update( int_slice(par_state, self.parallel_vars))
        #log.debug("SliceDict: {}", slice_dict)
        return slice_dict

