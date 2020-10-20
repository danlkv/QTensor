import copy
import time
import numpy as np
import itertools
import qtree
from qtree.optimizer import Var
import matplotlib.pyplot as plt
import networkx as nx
from tqdm.auto import tqdm
import operator

def get_neighbours_peo(old_graph):
    graph = copy.deepcopy(old_graph)
    graph.remove_edges_from(nx.selfloop_edges(old_graph))
    peo = []
    nghs = []

    while graph.number_of_nodes():
        ###start = time.time()
        costs = np.array(list(
            map(len, map(operator.itemgetter(1), graph.adjacency()))
        ))
        #costs = list(graph.degree)
        ###costtime = time.time() - start

        ###start = time.time()
        best_idx = np.argmin(costs)
        best_degree = costs[best_idx]
        best_node = list(graph.nodes())[best_idx]
        del costs
        peo.append(int(best_node))
        nghs.append(int(best_degree))
        #nodeiter_time = time.time() - start


        #start = time.time()
        #qtree.graph_model.eliminate_node(graph, best_node)
        eliminate_node_no_structure(graph, best_node)
        #eltime = time.time() - start
        #pbar.set_postfix(costiter=1/costtime, nodeiter=1/nodeiter_time, eliter=1/eltime ,costtime=costtime, nodeiter_time=nodeiter_time, eltime=eltime)
    return peo, nghs

def eliminate_node_no_structure(graph, node):
    neighbors_wo_node = set(graph[node])
    while node in neighbors_wo_node:
        neighbors_wo_node.remove(node)

    graph.remove_node(node)

    # prepare new tensor
    if len(neighbors_wo_node) > 1:
        graph.add_edges_from( itertools.combinations(neighbors_wo_node, 2))


def get_locale_peo(old_graph, rule):
    # This is far below computationally effective
    graph = copy.deepcopy(old_graph)
    
    path= []
    vals = []

    with tqdm(total=graph.number_of_nodes(), desc='Node removal') as pbar:
        while graph.number_of_nodes():
            #nodes = sorted(graph.nodes, key=int)
            nodes = sorted(list(graph.nodes), key=int)
            rule_ = lambda n: rule(graph, n)
            start = time.time()
            costs = list(map(rule_, nodes))
            costtime = time.time() - start
            _idx = np.argmin(costs)
            vals.append(costs[_idx])
            node = nodes[_idx]
            path.append(node)
            start = time.time()
            eliminate_node_no_structure(graph, node)
            eltime = time.time() - start
            pbar.update(1)
            pbar.set_postfix(eliter=1/eltime, costiter=1/costtime, degree=costs[_idx])
    return path, vals


def get_test_circ_filename(root, size, depth=10, id_=0):
    grid = f'{size}x{size}'
    return f'{root}/inst/cz_v2/{grid}/inst_{grid}_{depth}_{id_}.txt'

def reorder_graph(graph, peo):
    graph, label_dict = qtree.graph_model.relabel_graph_nodes(
        graph, dict(zip(peo, sorted(graph.nodes(), key=int)))
    )
    return graph, label_dict


def plot_cost(mems, flops):
    plt.yscale('log')
    ax = plt.gca()
    ax.grid(which='minor', alpha=0.5, linestyle='-', axis='both')
    ax.grid(which='major', alpha=0.6, axis='both')
    ax.yaxis.set_tick_params(labelbottom=True)
    #ax.minorticks_on()

    plt.plot(mems, label='Memory')
    plt.plot(flops, label='FLOP')
    #plt.legend()


def nodes_to_vars(old_graph, peo):
    peo_vars = [Var(v, size=old_graph.nodes[v]['size'],
                    name=old_graph.nodes[v]['name']) for v in peo]
    return peo_vars


def n_neighbors(graph, node):
    return len(list(graph[node].values()))

def degree(graph, node):
    return graph.degree(node)


def edges_to_clique(graph, node):
    N = graph.degree(node)
    edges = graph.edges(node)
    return N*(N-1)//2 - len(edges)


def _neighbors(graph, node):
    return list(graph.neighbors(node))


def get_neighbours_path(old_graph, peo=None):
    if peo is not None:
        graph, _ = reorder_graph(old_graph, peo)
    else:
        graph = copy.deepcopy(old_graph)
    ngh = []
    nodes = sorted(graph.nodes, key=int)
    for node in nodes:
        ngh.append(n_neighbors(graph, node))
        #qtree.graph_model.eliminate_node(graph, node)
        eliminate_node_no_structure(graph, node)
    return nodes, ngh

def nodes_at_distance(G, nodes, dist):
    nodes = list(nodes)
    for d in range(dist):
        range_d_nodes = []
        for n in nodes:
            neigh = list(G[n].keys())
            range_d_nodes += neigh
        nodes += range_d_nodes
        nodes  = list(set(nodes))
    return set(nodes)

def get_edge_subgraph(G, edge, dist):
    nodes = nodes_at_distance(G, edge, dist)
    return G.subgraph(set(nodes))


class ReportTable():
    measures = {
        'max':np.max
        ,'min': np.min
        ,'mean': np.mean
        ,'sum': np.sum
        ,'median': np.median
    }
    def __init__(self, measure=['max', 'min'], columns=[], max_records=100):
        self.measure = measure
        self.columns = columns
        self.records = []
        self.max_records = max_records

    def record(self, **kwargs):
        if self.columns:
            if set(self.columns) != set(kwargs.keys()):
                raise ValueError(f"columns doesn't match: {kwargs.keys()}, expect: {self.columns}")
        else:
            self.columns = set(kwargs.keys())
        self.records += [[kwargs[key] for key in self.columns]]

    def _title_row(self):
        return ['N'] + list(self.columns)

    def _print_title(self):
        print(','.join(self._title_row()))

    def _format_row(self, row):
        def format(x):
            if isinstance(x, str): return x
            if x == 0:
                return '0'
            if x > 1000 or x<0.001:
                return f"{x:.3e}"
            if isinstance(x, int):
                return str(x)
            return f"{x:.3f}"
        return [format(x) for x in row]

    def _measure_rows(self):
        rows = []
        for stat_label in self.measure:
            stat_func = self.measures[stat_label]
            stat_row = [stat_label]
            for i, key in enumerate(self.columns):
                column = [row[i] for row in self.records]
                stat_row.append(stat_func(column))
            rows.append(stat_row)
        return rows

    def _print_row(self, row):
        print(','.join(self._format_row(row)))

    def markdown(self):
        cells = []
        cells.append(self._title_row())
        for i, row in enumerate(self.records):
            cells.append([str(i)] + self._format_row(row))
        cells += [self._format_row(r) for r in self._measure_rows()]
        table = MarkdownTable(cells)
        return table.markdown()

    def print(self):
        self._print_title()
        for i in range(min(self.max_records, len(self.records))):
            self._print_row([i] + list(self.records[i]))

        for stat_row in self._measure_rows():
            self._print_row(stat_row)

class MarkdownTable:
    """ Stolen from https://github.com/lzakharov/csv2md/blob/master/csv2md/table.py """
    def __init__(self, cells):
        self.cells = cells
        self.widths = list(map(max, zip(*[list(map(len, row)) for row in cells])))

    def markdown(self, center_aligned_columns=None, right_aligned_columns=None):
        def format_row(row):
            return '| ' + ' | '.join(row) + ' |'

        rows = [format_row([cell.ljust(width) for cell, width in zip(row, self.widths)]) for row in self.cells]
        separators = ['-' * width for width in self.widths]

        if right_aligned_columns is not None:
            for column in right_aligned_columns:
                separators[column] = ('-' * (self.widths[column] - 1)) + ':'
        if center_aligned_columns is not None:
            for column in center_aligned_columns:
                separators[column] = ':' + ('-' * (self.widths[column] - 2)) + ':'

        rows.insert(1, format_row(separators))

        return '\n'.join(rows)

    @staticmethod
    def parse_csv(file, delimiter=',', quotechar='"'):
        return Table(list(csv.reader(file, delimiter=delimiter, quotechar=quotechar)))

