import numpy as np

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
