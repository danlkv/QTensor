from qtensor.MergedSimulator import MergedQAOASimulator, MergedQAOASimulatorSymmetryAccelerated, MergedSimulator
import qtree
import numpy as np
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm

import qtensor
from qtensor.contraction_backends import NumpyBackend
from qtensor.Simulate import Simulator, QtreeSimulator
from qtensor.optimisation.Optimizer import SlicesOptimizer
from qtensor.optimisation.TensorNet import QtreeTensorNet
import psutil

from loguru import logger as log

from qtensor import utils

def int_slice(value, vars_to_slice):
    """
    Creates a slice dict with integers an values.
    """
    dimensions = [var.size for var in vars_to_slice]
    multiindex = qtree.utils.unravel_index(value, dimensions)

    return {idx: val for idx, val in zip(vars_to_slice, multiindex)}

class FeynmanSimulator(QtreeSimulator):
    FallbackOptimizer = SlicesOptimizer

    def __init__(self, *args,
                 pool_type='process', n_processes=None
                 , **kwargs):
        super().__init__(*args, **kwargs)
        if n_processes is None:
            self.n_processes = 2
        else:
            self.n_processes = n_processes
        if pool_type == 'thread':
            self.pool = ThreadPool
        else:
            self.pool = Pool

    def optimize_buckets(self, fixed_vars: list=None):
        opt = self.optimizer
        peo, par_vars, self.tn = opt.optimize(self.tn)
        self.parallel_vars = par_vars
        return peo

    def _parallel_unit(self, par_idx):
        slice_dict = self._get_slice_dict(par_state=par_idx)

        sliced_buckets = self.tn.slice(slice_dict)
        result = qtree.optimizer.bucket_elimination(
            sliced_buckets, self.backend.process_bucket
        , n_var_nosum=len(self.tn.free_vars + self.parallel_vars))

        return self.backend.get_result_data(result).flatten()

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

        with self.pool(self.n_processes) as p:
            total_paths = 2**len(self.parallel_vars)
            log.info('Starting to simulate {} paths using {} processes', total_paths, self.n_processes)
            args = range(total_paths)
            piter = p.imap(self._parallel_unit, args)
            r = list(tqdm(piter, total=total_paths))
            #r = list(piter)
        result = sum(r)
        return result

    def _reorder_buckets(self):
        perm_dict = super()._reorder_buckets()
        self.parallel_vars = sorted([perm_dict[idx] for idx in self.parallel_vars], key=str)
        return perm_dict

    def _get_slice_dict(self, initial_state=0, target_state=0, par_state=0):
        slice_dict = super()._get_slice_dict(initial_state, target_state)
        slice_dict.update( int_slice(par_state, self.parallel_vars))
        #log.debug("SliceDict: {}", slice_dict)
        return slice_dict

class FeynmanMergedSimulator(FeynmanSimulator, MergedSimulator):
    def _convert_peo_raw(self, peo):
        """
        Process input peo

        if ``peo'' is None than find the ordering
        then relabel it to use idintities from curent buckets
        """
        if peo is None:
            opt = self.optimizer
            peo, par_vars, self.tn = opt.optimize(self.tn)
            self.parallel_vars = par_vars
            if self.max_tw:
                if self.optimizer.treewidth > self.max_tw:
                    raise ValueError(f'Treewidth {self.optimizer.treewidth} is larger than max_tw={self.max_tw}.')

        all_indices = self._all_bucket_indices()
        identity_map = {v.name: v for v in all_indices}
        self.parallel_vars = [identity_map[i.name] for i in self.parallel_vars]
        return [identity_map[i.name] for i in peo]

    def simulate(self, qc, batch_vars=0, tw_bias=2, peo=None,
                 #dry_run=False
                 ):
        self.tw_bias = tw_bias
        self._new_circuit(qc)
        self._create_buckets()
        free_final_qubits = list(range(batch_vars))
        self._set_free_qubits(free_final_qubits)
        self.peo = self._convert_peo_raw(peo)
        perm_dict = self._reorder_buckets()
        #-- Merged ix handling
        # Get tensor indices for merging subroutine
        bucket_ix = [[set(t.indices) for t in bucket]
                     for bucket in self.tn.buckets
                    ]
        merged_ix, width = qtensor.utils.find_mergeable_indices(self.peo, bucket_ix)

        self._merge_buckets(merged_ix)
        #--


        with self.pool(self.n_processes) as p:
            total_paths = 2**len(self.parallel_vars)
            log.info('Starting to simulate {} paths using {} processes', total_paths, self.n_processes)
            args = range(total_paths)
            piter = p.imap(self._parallel_unit, args)
            r = list(tqdm(piter, total=total_paths))
            #r = list(piter)
        result = sum(r)
        return result

    def _parallel_unit(self, par_idx):
        slice_dict = self._get_slice_dict(par_state=par_idx)
        # remove parallel vars from peo, they will not we in sliced buckets
        self.peo = [x for x in self.peo if x not in self.parallel_vars]

        identity_map = self._slice_buckets(slice_dict)
        # remove parallel vars from ibunch, they will are not in the sliced buckets
        self.ibunch = [[identity_map[int(y)] for y in x if y not in self.parallel_vars] for x in self.ibunch]
        #self.parallel_vars = [identity_map[x] for x in self.parallel_vars]
        #-- 
        # A dirty workaround to pass the merged buckets to benchmark optimizaiton code
        # TODO: decompose the functions to be able to get buckets
        self.merged_buckets = self.tn.buckets
        self.ibunch = self.ibunch

        result = qtensor.merged_indices.bucket_elimination(
            self.tn.buckets,
            self.ibunch,
            self.backend.process_bucket_merged,
            n_var_nosum=len(self.tn.free_vars)
        )
        #--
        return self.backend.get_result_data(result).flatten()
