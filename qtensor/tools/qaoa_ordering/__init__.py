from networkx.classes.function import number_of_nodes
import networkx as nx
import qtensor

def get_graph_order(graph: nx.Graph, opt_key:str):
    opt = qtensor.toolbox.get_ordering_algo(opt_key)
    peo, path = opt._get_ordering_ints(graph) #
    return peo


def circ2gvars(qubit_count, circuit, pdict={},
               omit_terminals=True):
    """
    Constructs a graph from a circuit in the form of a
    list of lists.

    Parameters
    ----------
    qubit_count : int
            number of qubits in the circuit
    circuit : list of lists
            quantum circuit as returned by
            :py:meth:`operators.read_circuit_file`
    pdict : dict
            Dictionary with placeholders if any parameteric gates
            were unresolved
    max_depth : int, default None
            Maximal depth of the circuit which should be used
    omit_terminals : bool, default True
            If terminal nodes should be excluded from the final
            graph.

    Returns
    -------
        gvars: dict(int: list)
    """
    import functools, itertools
    import qtree.operators as ops
    from qtree.optimizer import Var

    # The circuit is built from left to right, as it operates
    # on the ket ( |0> ) from the left. We thus first place
    # the bra ( <x| ) and then put gates in the reverse order

    # Fill the variable `frame`
    layer_variables = list(range(qubit_count))
    result = {i: [v] for i, v in enumerate(layer_variables)}
    current_var_idx = qubit_count

    # Populate nodes and save variables of the bra
    bra_variables = []
    for var in layer_variables:
        bra_variables.append(Var(var, name=f"o_{var}"))

    # Place safeguard measurement circuits before and after
    # the circuit
    measurement_circ = [[ops.M(qubit) for qubit in range(qubit_count)]]

    combined_circ = functools.reduce(
        lambda x, y: itertools.chain(x, y),
        [measurement_circ, reversed(circuit)])

    # Start building the graph in reverse order
    for layer in combined_circ:
        for op in layer:
            # Update current variable frame
            for qubit in op.changed_qubits:
                new_var = Var(current_var_idx, name=f"o_{current_var_idx}")
                result[qubit].append(new_var)
                current_var_idx += 1

    for qubit in range(qubit_count):
        var = layer_variables[qubit]
        new_var = Var(current_var_idx, name=f'i_{qubit}', size=2)
        # update graph and variable `frame`
        result[qubit].append(new_var)
        current_var_idx += 1

    if omit_terminals:
        for k in result:
            # first and last elements will always be terminals
            result[k] = result[k][1:-1]

    return result


def build_connectivity_graph(circuit):
    g = nx.Graph()
    for gate in circuit:
        if len(gate.qubits) == 2:
            # will add muliple edges several times, but who cares
            g.add_edge(gate.qubits[0], gate.qubits[1])
    return g


def get_qaoa_exp_ordering(circuit: list, algo='greedy') -> list:
    graph = build_connectivity_graph(circuit)
    cverts = circ2gvars(graph.number_of_nodes(), [circuit])
    super_order = get_graph_order(graph, algo)
    order = []
    for i in super_order:
        order += cverts[i]
    return order


class QAOAEnergyOptimizer(qtensor.optimisation.GreedyOptimizer):
    # debt: this class is only usable for a single instance
    def __init__(self, circuit, *args, algo='greedy', **kwargs):
        super().__init__(*args, **kwargs)
        self.circuit = circuit
        self.algo = algo

    def _get_ordering_ints(self, old_graph, free_vars):
        return get_qaoa_exp_ordering(self.circuit, algo=self.algo)