import qtree
import psutil
import numpy as np

from qensor import utils
from qensor.optimisation.Greedy import GreedyParvars
from loguru import logger as log


class Optimizer:
    def optimize(self, tensor_net):
        raise NotImplementedError

class OrderingOptimizer(Optimizer):
    def _get_ordering_ints(self, graph, fixed_vars=[]):
        peo_ints, path = utils.get_locale_peo(graph, utils.n_neighbors)

        return peo_ints, path

    def optimize(self, tensor_net):
        line_graph = tensor_net.get_line_graph()
        fixed_vars = tensor_net.fixed_vars
        ignored_vars = tensor_net.ket_vars + tensor_net. bra_vars
        graph = line_graph

        if fixed_vars:
            graph = qtree.graph_model.make_clique_on(graph, fixed_vars)

        peo, path = self._get_ordering_ints(graph)
        self.treewidth = max(path)

        peo = [qtree.optimizer.Var(var, size=graph.nodes[var]['size'],
                        name=graph.nodes[var]['name'])
                    for var in peo]
        if fixed_vars:
            peo = qtree.graph_model.get_equivalent_peo(graph, peo, fixed_vars)

        peo = ignored_vars + peo
        self.peo = peo
        self.ignored_vars = ignored_vars
        return peo, tensor_net



class SlicesOptimizer(OrderingOptimizer):

    def __init__(self, tw_bias=2):
        self.tw_bias = tw_bias

    def _get_max_tw(self):
        mem = psutil.virtual_memory()
        avail = mem.available
        log.info('Memory available: {}', avail)
        # Cost = 16*2**tw
        # tw = log(cost/16) = log(cost) - 4
        return np.int(np.log2(avail)) - 4

    def _split_graph(self, p_graph, max_tw):
        searcher = GreedyParvars(p_graph)
        peo_ints, path = self._get_ordering_ints(p_graph)
        while True:
            #nodes, path = utils.get_neighbours_path(graph, peo=peo_ints)
            tw = self.treewidth
            log.info('Treewidth: {}', tw)
            if tw < max_tw:
                log.info('Found parvars: {}', searcher.result)
                break
            error = searcher.step()
            pv_cnt = len(searcher.result)
            log.debug('Parvars count: {}. Amps count: {}', pv_cnt, 2**pv_cnt)
            if error:
                log.error('Memory is not enough. Max tw: {}', max_tw)
                raise Exception('Estimated OOM')

            peo_ints, path = self._get_ordering_ints(p_graph)
            self.treewidth = max(path)

        return peo_ints, searcher.result

    def optimize(self, tensor_net):
        peo, tensor_net = super().optimize(tensor_net)
        graph = tensor_net.get_line_graph()

        p_graph = graph.copy()
        max_tw = self._get_max_tw()
        log.info('Maximum treewidth: {}', max_tw)
        max_tw = max_tw - self.tw_bias

        peo, par_vars = self._split_graph(p_graph, max_tw)

        # TODO: move these platform-dependent things
        self.parallel_vars = [
            qtree.optimizer.Var(var,
                                size=graph.nodes[var]['size'],
                                name=graph.nodes[var]['name'])
                              for var in par_vars]
        peo = [qtree.optimizer.Var(var, size=graph.nodes[var]['size'],
                        name=graph.nodes[var]['name'])
                    for var in peo]

        self.peo = self.ignored_vars + peo + self.parallel_vars 
        log.info('peo {}', self.peo)
        return self.peo, self.parallel_vars, tensor_net
