import numpy as np
import copy, operator
import time
from qtensor.optimisation.Optimizer import GreedyOptimizer
from qtensor.optimisation.networkit import greedy_ordering_networkit
from qtensor import utils
from functools import reduce
import networkx as nx
import qtree

def reducelist(f, lst, x=0):
    prev = x
    for i in lst:
        prev = f(prev, i)
        yield prev

class RGreedyOptimizer(GreedyOptimizer):
    """
    An orderer that greedy selects vertices
    using boltzman probabilities.

    """
    def __init__(self, *args, temp=0.002, repeats=10,
                 max_time=np.inf,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.temp = temp
        self.repeats = repeats
        self.max_time = max_time

    def _get_ordering(self, graph, **kwargs):
        #mapping = {i:k for i, k in enumerate(graph.nodes)}
        #graph = nx.convert_node_labels_to_integers(graph)
        node_names = nx.get_node_attributes(graph, 'name')
        node_sizes = nx.get_node_attributes(graph, 'size')
        peo, path = self._get_ordering_ints(graph)

        peo = [qtree.optimizer.Var(var, size=node_sizes[var],
                        name=node_names[var])
                    for var in peo]
        #print('tw=', max(path))
        return peo, path

    def _get_ordering_ints(self, old_graph, free_vars=[]):
        best_peo = None
        best_width = np.inf
        best_widths = None
        start_time = time.time()

        for i in range(self.repeats):
            graph = copy.deepcopy(old_graph)
            peo = []
            widths = []
            while graph.number_of_nodes():
                ngs = np.array(list(
                    map(len, map(operator.itemgetter(1), graph.adjacency()))
                ))

                weights = np.exp(-(ngs - np.min(ngs))/self.temp)
                #print(ngs)
                #print(weights)
                # 1, 3, 5, 2, 1
                distrib = np.array([0]+list(reducelist(lambda x, y:x+y, weights, 0)))
                #print(distrib)
                # 0, 1, 4, 9, 11, 12
                rnd = np.random.random()*distrib[-1]
                # between 0 and 12  = say, 5
                # find the smallest value that larger than rnd
                bool_map = distrib < rnd
                # True, True, True, False, False, False
                select_map = bool_map[1:] ^ bool_map[:-1]
                selected_elem = np.array(list(graph.nodes))[select_map]
                assert len(selected_elem)==1, 'Error in algorithm, please submit an issue'
                selected_node = selected_elem[0]
                utils.eliminate_node_no_structure(graph, selected_node)

                peo.append(int(selected_node))
                widths.append(int(ngs[select_map][0]))

            if max(widths) < best_width:
                best_peo = peo
                best_widths = widths
                best_width = max(widths)

            if time.time() - start_time > self.max_time:
                break

        return best_peo, best_widths

class RGreedyOptimizerNk(RGreedyOptimizer):

    def _get_ordering_ints(self, old_graph, free_vars=[]):
        best_peo = None
        best_width = np.inf
        best_widths = None
        start_time = time.time()

        for i in range(self.repeats):
            graph = copy.deepcopy(old_graph)
            peo = []
            widths = []
            while graph.number_of_nodes():
                ngs = np.array(list(
                    map(len, map(operator.itemgetter(1), graph.adjacency()))
                ))

                weights = np.exp(-(ngs - np.min(ngs))/self.temp)
                #print(ngs)
                #print(weights)
                # 1, 3, 5, 2, 1
                distrib = np.array([0]+list(reducelist(lambda x, y:x+y, weights, 0)))
                #print(distrib)
                # 0, 1, 4, 9, 11, 12
                rnd = np.random.random()*distrib[-1]
                # between 0 and 12  = say, 5
                # find the smallest value that larger than rnd
                bool_map = distrib < rnd
                # True, True, True, False, False, False
                select_map = bool_map[1:] ^ bool_map[:-1]
                selected_elem = np.array(list(graph.nodes))[select_map]
                assert len(selected_elem)==1, 'Error in algorithm, please submit an issue'
                selected_node = selected_elem[0]
                utils.eliminate_node_no_structure(graph, selected_node)

                peo.append(int(selected_node))
                widths.append(int(ngs[select_map][0]))

            if max(widths) < best_width:
                best_peo = peo
                best_widths = widths
                best_width = max(widths)

            if time.time() - start_time > self.max_time:
                break

        return best_peo, best_widths
