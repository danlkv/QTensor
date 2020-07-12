import qtree
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

from qensor.ProcessingFrameworks import NumpyBackend
from qensor.Simulate import Simulator, QtreeSimulator
import psutil

from loguru import logger as log

from qensor import utils

class GreedyOpt:
    """
    iterable:
        the items to pick from

    size:
        size of subset of items to search

    target:
        Function to minimize
    """

    def __init__(self, iterable=[], target=lambda x: 1):
        self.iterable = iterable
        self._target = target
        self.result = []
        self.min_vals = []
        self.min_items = []

    def set_target(self, target):
        self._target = target

    def target(self, item):
        """
        Called every search len(iterable) times.

        Total number of calls: size*items
        """
        return self._target(item)

    def add(self, item):
        """
        called every time a minimum found

        Total number of calls: size
        """

        self.result.append(item)

    def run(self, size):
        return self.run_size(size)

    def items(self):
        return self.iterable

    def step(self):
        items = np.array(self.items())
        costs = np.array([self.target(i) for i in items])
        if len(costs) == 0:
            return 1

        min_idx = np.argmin(costs)
        min_item = items[min_idx]
        min_val = costs[min_idx]

        self.min_items.append(min_item)
        self.min_vals.append(min_val)
        self.add(min_item)


    def run_cost(self, cost):
        while True:
            error_code = self.step()
            if error_code==1:
                print('Greedy search failed to find desired cost')
                raise Exception('Failed to optimize')
            if self.min_vals[-1] < cost:
                break

    def run_size(self, size):
        for i in range(size):
            self.step()

        return self.result

class GreedyParvars(GreedyOpt):
    def __init__(self, graph, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = graph

    def items(self):
        return self.graph.nodes

    def target(self, item):
        return - self.graph.degree(item)

    def add(self, item):
        super().add(item)
        self.graph.remove_node(item)
        #qtree.graph_model.eliminate_node(self.graph, item)


class FeynmanSimulator(QtreeSimulator):

    def _get_max_tw(self):
        mem = psutil.virtual_memory()
        avail = mem.available
        log.info('Memory available: {}', avail)
        # Cost = 16*2**tw
        # tw = log(cost/16) = log(cost) - 4
        return np.int(np.log2(avail)) - 4


    def optimize_buckets(self, buckets, ignored_vars=[], fixed_vars: list=None):
        orig_graph = qtree.graph_model.buckets2graph(buckets,
                                               ignore_variables=ignored_vars)
        if fixed_vars:
            cl_graph = qtree.graph_model.make_clique_on(orig_graph, fixed_vars)

        graph = cl_graph.copy()
        searcher = GreedyParvars(graph)
        max_tw = self._get_max_tw()
        log.info('Maximum treewidth: {}', max_tw)
        peo_cl, path = utils.get_locale_peo(graph, utils.n_neighbors)
        peo_ints = peo_cl
        while True:
            #nodes, path = utils.get_neighbours_path(graph, peo=peo_ints)
            tw = max(path)
            log.info('Treewidth: {}', tw)
            if tw < max_tw - 3:
                log.info('Found parvars: {}', searcher.result)
                break
            error = searcher.step()
            pv_cnt = len(searcher.result)
            log.debug('Parvars count: {}. Amps count: {}', pv_cnt, 2**pv_cnt)
            if error:
                log.error('Memory is not enough. Max tw: {}', max_tw)
                raise Exception('Estimated OOM')

            peo_ints, path = utils.get_locale_peo(graph, utils.n_neighbors)
            peo_cl =  peo_ints + searcher.result


        self.treewidth = max(path)

        graph = cl_graph
        self.parallel_vars = [
            qtree.optimizer.Var(var,
                                size=graph.nodes[var]['size'],
                                name=graph.nodes[var]['name'])
                              for var in searcher.result]
        peo = [qtree.optimizer.Var(var, size=graph.nodes[var]['size'],
                        name=graph.nodes[var]['name'])
                    for var in peo_cl]
        if fixed_vars:
            peo = qtree.graph_model.get_equivalent_peo(graph, peo, fixed_vars)

        peo = ignored_vars + peo
        self.peo = peo
        return peo

    def _parallel_unit(self, par_idx):
        slice_dict = self._get_slice_dict(par_state=par_idx)

        sliced_buckets = self.bucket_backend.get_sliced_buckets(
            self.buckets, self.data_dict, slice_dict)
        result = qtree.optimizer.bucket_elimination(
            sliced_buckets, self.bucket_backend.process_bucket
        , n_var_nosum=len(self.free_bra_vars))
        return result.data.flatten()

    def simulate(self, qc, batch_vars=0):
        return self.simulate_batch_adaptive(qc, batch_vars)


    def simulate_batch_adaptive(self, qc, batch_vars=0):
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
        perm_buckets, perm_dict = qtree.optimizer.reorder_buckets(self.buckets, self.peo)
        self.ket_vars = sorted([perm_dict[idx] for idx in self.ket_vars], key=str)
        self.bra_vars = sorted([perm_dict[idx] for idx in self.bra_vars], key=str)
        self.parallel_vars = sorted([perm_dict[idx] for idx in self.parallel_vars], key=str)
        self.buckets = perm_buckets

    def _get_slice_dict(self, initial_state=0, target_state=0, par_state=0):
        slice_dict = qtree.utils.slice_from_bits(initial_state, self.ket_vars)
        slice_dict.update( qtree.utils.slice_from_bits(target_state, self.bra_vars))
        slice_dict.update({var: slice(None) for var in self.free_bra_vars})
        slice_dict.update( qtree.utils.slice_from_bits(par_state, self.parallel_vars))
        return slice_dict

