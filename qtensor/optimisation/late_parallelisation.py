"""
The idea of late parallelization is that we don't have to
slice our tensor network in the beginning of the contraction.

"""
import psutil
import numpy as np
from functools import partial
from loguru import logger as log
import qtree

import qtensor as qtn
from qtensor.optimisation import Optimizer, RGreedyOptimizer
from qtensor.optimisation import GreedyParvars

def slice_greedy(graph, p_bunch, ordering_algo='greedy'):
    """ Slice greedy and inplece """
    orderer = qtn.toolbox.get_ordering_algo(ordering_algo)
    searcher = GreedyParvars(graph)
    peo_ints, path = orderer._get_ordering_ints(graph)
    for _ in range(p_bunch):
        error = searcher.step()
        pv_cnt = len(searcher.result)
        log.debug('Parvars count: {}. Amps count: {}', pv_cnt, 2**pv_cnt)
        if error:
            raise Exception('Estimated OOM')
    return searcher.result


class LateParOptimizer(Optimizer):
    def __init__(self, target_tw=None, par_vars=None,
                 p_bunch = None,
                 n_bunches=None,
                 ordering_algo='greedy', slicing_algo='greedy'):
        """
        The optimizer works in the following way:
            1. Find ordering with provided optimizer
            2. For each step:
                3. Contract graph up to the step;
                4. Find p_bunch indices to slice;
                5. Find new ordering;
            6. Save the step with best performance.

        Args:
            target_tw (int): Slice until reached this tw.
                If None, use system memory to estimate.
                Defaults to None.

            par_vars (int): number of parallel vars to split. Overrides target_tw.
            n_bunches: How many bunches to slice. Overrides target_tw if not None.

        """
        self.orderer = qtn.toolbox.get_ordering_algo(ordering_algo)
        if target_tw is None:
            self.target_tw = self._get_max_tw()
        self.target_tw = target_tw

        self.n_bunches = n_bunches
        self.par_vars = par_vars

        if not n_bunches:
            self.p_bunch = 1
            self.n_bunches = par_vars
            #self.n_bunches = par_vars
        else:
            if p_bunch is None:
                self.p_bunch = par_vars//n_bunches
            else:
                self.p_bunch = p_bunch
        if slicing_algo == 'greedy':
            self.slicer = partial(slice_greedy, ordering_algo=ordering_algo)
        else:
            raise ValueError(f'Invalid slicing algorithm: {slicing_algo}')

    def _get_max_tw(self):
        if hasattr(self, 'max_tw') and self.max_tw is not None:
            return self.max_tw
        mem = psutil.virtual_memory()
        avail = mem.available
        log.info('Memory available: {}', avail)
        # Cost = 16*2**tw
        # tw = log(cost/16) = log(cost) - 4
        return np.int(np.log2(avail)) - 4

    def find_slice_at_step(self, ordering, graph, p_bunch):
        """
        Scaling:
            O(n*(Slicer(n)+Ordering(n))) where n is the number of nodes in the graph.
            O(2n^2) for greedy

        Returns:
            graph: sliced graph
            p_vars: parallel_vars
            step: index in ordering at which to slice
            peo: peo after slice
            treewidth: treewidth after slice
        """

        slice_candidates = []
        largest_tw = 0

        for node in ordering[:-p_bunch]:
            # Room for optimization: stop earlier
            # Room for optimization: do not copy graph
            sliced_graph = graph.copy()
            slice_vars = self.slicer(sliced_graph, p_bunch=p_bunch)
            _peo, _path = self.orderer._get_ordering_ints(sliced_graph)
            step_tw = qtn.utils.n_neighbors(graph, node) + 1
            largest_tw = max(step_tw, largest_tw)
            _tw = max(largest_tw, max(_path))
            slice_candidates.append(
                (slice_vars, _peo, _tw, sliced_graph)
            )
            qtn.utils.eliminate_node_no_structure(graph, node)

        slice_vars, peos, tws, graphs = zip(*slice_candidates)
        best_steps, *_ = np.where(tws == np.min(tws))
        best_step = best_steps[0]
        best_peo = peos[best_step]
        best_tw = tws[best_step]
        assert len(ordering[:best_step]) + len(best_peo) + p_bunch == len(ordering), \
                f"Invalid ordering: total nodes: {len(ordering)}," \
                f" step: {best_step}, len next_peo: {len(best_peo)}"
        return graphs[best_step], slice_vars[best_step], best_step, best_peo, best_tw


    def optimize(self, tensor_net):
        """
        Args:
            (qtensor.TensorNet): Tensor network to optimize
        Returns:
            parallel_scheme list((contraction_order, slice_vars)):
                Map from parallel var to step at which it is removed
        Mutates:
            self.treewidth
        """


        line_graph = tensor_net.get_line_graph()
        free_vars = tensor_net.free_vars
        ignored_vars = tensor_net.ket_vars + tensor_net. bra_vars

        if free_vars:
            current_graph = qtree.graph_model.make_clique_on(line_graph, free_vars)
        else:
            current_graph = line_graph

        current_ordering, tw_path = self.orderer._get_ordering_ints(current_graph)
        contraction_schedule = []
        log.info(f"Initial treewidth: {max(tw_path)}")

        # --
        if self.n_bunches is not None:
            # Iterate for fixed par_vars
            self.target_tw = 0
            bunches = [self.par_vars//self.n_bunches]*self.n_bunches
            _remaining = self.par_vars%self.n_bunches
            bunches = bunches + [_remaining]
            bunches = [x for x in bunches if x != 0]
        else:
            # Iterate until reach target_tw
            n_iter = len(current_ordering)
            bunches = [self.p_bunch]*n_iter
        # --
        for p_bunch in bunches:
            _a_bunch_of_stuff = self.find_slice_at_step(
                current_ordering, current_graph, p_bunch
            )
            current_graph, slice_vars, step, next_ordering, next_tw = _a_bunch_of_stuff
            contraction_schedule.append(
                (current_ordering[:step], slice_vars)
            )
            current_ordering = [x for x in next_ordering if x not in slice_vars]
            log.info(f"Sliced {len(slice_vars)}, next treewidth: {next_tw}")

            if next_tw <= self.target_tw:
                break

        log.info(f"Removed {sum(bunches)} variables, reduced tw by {max(tw_path)-next_tw}")
        # Contract leftovers
        if free_vars:
            if not all(x in current_ordering for x in free_vars):
                log.warning(f"Not all free variables are in the last ordering chunk!")
            current_ordering = qtree.graph_model.get_equivalent_peo(
                current_graph, current_ordering, free_vars
            )

            next_tw_eq = qtree.graph_model.get_treewidth_from_peo(
                current_graph, current_ordering
            )
            assert next_tw == next_tw_eq

        self.treewidth = next_tw




        contraction_schedule.append((current_ordering, tuple()))
        first_slice, first_ordering = contraction_schedule[0]
        first_ordering = ignored_vars + first_ordering
        contraction_schedule[0] = (first_slice, first_ordering)
        return contraction_schedule

