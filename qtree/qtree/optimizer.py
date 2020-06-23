"""
Operations to load/contract quantum circuits. All functions
operating on Buckets (without any specific framework) should
go here.
"""

import functools
import itertools
import random
import networkx as nx
import qtree.operators as ops

from qtree.logger_setup import log

random.seed(0)


class Var(object):
    """
    Index class. Primarily used to store variable id:size pairs
    """
    def __init__(self, identity, size=2, name=None):
        """
        Initialize the variable
        identity: int
              Index identifier. We use mainly integers here to
              make it play nicely with graphical models.
        size: int, optional
              Size of the index. Default 2
        name: str, optional
              Optional name tag. Defaults to "v[{identity}]"
        """
        self._identity = identity
        self._size = size
        if name is None:
            name = f"v_{identity}"
        self._name = name
        self.__hash = hash((identity, name, size))

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return self._size

    @property
    def identity(self):
        return self._identity

    def copy(self, identity=None, size=None, name=None):
        if identity is None:
            identity = self._identity
        if size is None:
            size = self._size

        return Var(identity, size, name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def __int__(self):
        return int(self.identity)

    def __hash__(self):
        return self.__hash

    def __eq__(self, other):
        return (isinstance(other, type(self))
                and self.identity == other.identity
                and self.size == other.size
                and self.name == other.name)

    def __lt__(self, other):  # this is required for sorting
        return self.identity < other.identity


class Tensor(object):
    """
    Placeholder tensor class. We use it to do manipulations of
    tensors kind of symbolically and to not move around numpy arrays
    """
    def __init__(self, name, indices,
                 data_key=None, data=None):
        """
        Initialize the tensor
        name: str,
              the name of the tensor. Used only for display/convenience.
              May be not unique.
        indices: tuple,
              Indices of the tensor
        shape: tuple,
              shape of a tensor
        data_key: int
              Key to find tensor's data in the global storage
        data: np.array
              Actual data of the tensor. Default None.
              Usually is not supplied at initialization.
        """
        self._name = name
        self._indices = tuple(indices)
        self._data_key = data_key
        self._data = data
        self._order_key = hash((self.data_key, self.name))

    @property
    def name(self):
        return self._name

    @property
    def indices(self):
        return self._indices

    @property
    def shape(self):
        return tuple(idx.size for idx in self._indices)

    @property
    def data_key(self):
        return self._data_key

    @property
    def data(self):
        return self._data

    def copy(self, name=None, indices=None, data_key=None, data=None):
        if name is None:
            name = self.name
        if indices is None:
            indices = self.indices
        if data_key is None:
            data_key = self.data_key
        if data is None:
            data = self.data
        return Tensor(name, indices, data_key, data)

    def __str__(self):
        return '{}({})'.format(self._name, ','.join(
            map(str, self.indices)))

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        return self._order_key < other._order_key

    def __mul__(self, other):
        if self._data is None:
            raise ValueError(f'No data assigned in tensor {self.name}')
        if self.indices == other.indices:
            return self.copy(data=self._data * other._data)
        else:
            raise ValueError(f'Index mismatch in __mul__: {self.indices} times {other.indices}')

    def __eq__(self, other):
        return (isinstance(other, type(self))
                and self.name == other.name
                and self.indices == other.indices
                and self.data_key == other.data_key
                and self.data == other.data)


def circ2buckets(qubit_count, circuit, pdict={}, max_depth=None):
    """
    Takes a circuit in the form of list of lists, builds
    corresponding buckets. Buckets contain Tensors
    defining quantum gates. Each bucket
    corresponds to a variable. Each bucket can hold tensors
    acting on it's variable and variables with higher index.

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

    max_depth : int
            Maximal depth of the circuit which should be used
    Returns
    -------
    buckets : list of lists
            list of lists (buckets)
    data_dict : dict
            Dictionary with all tensor data
    bra_variables : list
            variables of the output qubits
    ket_variables: list
            variables of the input qubits
    """
    # import pdb
    # pdb.set_trace()

    if max_depth is None:
        max_depth = len(circuit)

    data_dict = {}

    # Let's build buckets for bucket elimination algorithm.
    # The circuit is built from left to right, as it operates
    # on the ket ( |0> ) from the left. We thus first place
    # the bra ( <x| ) and then put gates in the reverse order

    # Fill the variable `frame`
    layer_variables = [Var(qubit, name=f'o_{qubit}')
                       for qubit in range(qubit_count)]
    current_var_idx = qubit_count

    # Save variables of the bra
    bra_variables = [var for var in layer_variables]

    # Initialize buckets
    for qubit in range(qubit_count):
        buckets = [[] for qubit in range(qubit_count)]

    # Place safeguard measurement circuits before and after
    # the circuit
    measurement_circ = [[ops.M(qubit) for qubit in range(qubit_count)]]

    combined_circ = functools.reduce(
        lambda x, y: itertools.chain(x, y),
        [measurement_circ, reversed(circuit[:max_depth])])

    # Start building the graph in reverse order
    for layer in combined_circ:
        for op in layer:
            # build the indices of the gate. If gate
            # changes the basis of a qubit, a new variable
            # has to be introduced and current_var_idx is increased.
            # The order of indices
            # is always (a_new, a, b_new, b, ...), as
            # this is how gate tensors are chosen to be stored
            variables = []
            current_var_idx_copy = current_var_idx
            min_var_idx = current_var_idx
            for qubit in op.qubits:
                if qubit in op.changed_qubits:
                    variables.extend(
                        [layer_variables[qubit],
                         Var(current_var_idx_copy)])
                    current_var_idx_copy += 1
                else:
                    variables.extend([layer_variables[qubit]])
                min_var_idx = min(min_var_idx,
                                  int(layer_variables[qubit]))

            # fill placeholders in parameters if any
            for par, value in op.parameters.items():
                if isinstance(value, ops.placeholder):
                    op._parameters[par] = pdict[value]

            data_key = (op.name,
                        hash((op.name, tuple(op.parameters.items()))))
            # Build a tensor
            t = Tensor(op.name, variables,
                       data_key=data_key)

            # Insert tensor data into data dict
            data_dict[data_key] = op.gen_tensor()

            # Append tensor to buckets
            # first_qubit_var = layer_variables[op.qubits[0]]
            buckets[min_var_idx].append(t)

            # Create new buckets and update current variable frame
            for qubit in op.changed_qubits:
                layer_variables[qubit] = Var(current_var_idx)
                buckets.append(
                    []
                )
                current_var_idx += 1

    # Finally go over the qubits, append measurement gates
    # and collect ket variables
    ket_variables = []

    op = ops.M(0)  # create a single measurement gate object
    data_key = (op.name, hash((op.name, tuple(op.parameters.items()))))
    data_dict.update(
        {data_key: op.gen_tensor()})

    for qubit in range(qubit_count):
        var = layer_variables[qubit]
        new_var = Var(current_var_idx, name=f'i_{qubit}', size=2)
        ket_variables.append(new_var)
        # update buckets and variable `frame`
        buckets[int(var)].append(
            Tensor(op.name,
                   indices=[var, new_var],
                   data_key=data_key)
        )
        buckets.append([])
        layer_variables[qubit] = new_var
        current_var_idx += 1

    return buckets, data_dict, bra_variables, ket_variables


def bucket_elimination(buckets, process_bucket_fn,
                       n_var_nosum=0):
    """
    Algorithm to evaluate a contraction of a large number of tensors.
    The variables to contract over are assigned ``buckets`` which
    hold tensors having respective variables. The algorithm
    proceeds through contracting one variable at a time, thus we eliminate
    buckets one by one.

    Parameters
    ----------
    buckets : list of lists
    process_bucket_fn : function
              function that will process this kind of buckets
    n_var_nosum : int, optional
              number of variables that have to be left in the
              result. Expected at the end of bucket list
    Returns
    -------
    result : numpy.array
    """
    # import pdb
    # pdb.set_trace()
    n_var_contract = len(buckets) - n_var_nosum

    result = None
    for n, bucket in enumerate(buckets[:n_var_contract]):
        if len(bucket) > 0:
            tensor = process_bucket_fn(bucket)
            if len(tensor.indices) > 0:
                # tensor is not scalar.
                # Move it to appropriate bucket
                first_index = int(tensor.indices[0])
                buckets[first_index].append(tensor)
            else:   # tensor is scalar
                if result is not None:
                    result *= tensor
                else:
                    result = tensor

    # form a single list of the rest if any
    rest = list(itertools.chain.from_iterable(buckets[n_var_contract:]))
    if len(rest) > 0:
        # only multiply tensors
        tensor = process_bucket_fn(rest, no_sum=True)
        if result is not None:
            result *= tensor
        else:
            result = tensor
    return result


def graph2buckets(graph):
    """
    Takes a Networkx MultiGraph and produces a corresponding
    bucket list. This is an inverse of the :py:meth:`buckets2graph`

    Parameters
    ----------
    graph : networkx.MultiGraph
            contraction graph of the circuit. Has to support self loops
            and parallel edges. Parallel edges are needed to support
            multiple qubit operators on same qubits
            (which can be collapsed in one operation)

    Returns
    -------
    buckets : list of lists
    """
    buckets = []

    # import pdb
    # pdb.set_trace()
    variables = sorted(graph.nodes(data=False))

    # Add buckets with sorted variables
    for variable in variables:
        # First collect all unique tensors (they may be elements of
        # the current bucket)
        # go over edges (pairs of variables).
        # The first variable in pair is this variable
        candidate_tensors = {}

        for edge in graph.edges(variables, data=True):
            _, other_variable, edge_data = edge
            tensor = edge_data['tensor']

            key = (tensor['name'], tensor['indices'], tensor['data_key'])
            if key not in candidate_tensors:
                # turn integers into Var objects
                indices_vars = tuple(Var(var,
                                         name=graph.nodes[var]['name'],
                                         size=graph.nodes[var]['size'])
                                     for var in tensor['indices'])
                # Form Tensor objects and
                # place candidate tensors into hash table
                candidate_tensors[key] = (
                    Tensor(
                        name=tensor['name'],
                        indices=indices_vars,
                        data_key=tensor['data_key']
                    )
                )

        # Now we have all tensors in bucket format.
        # Drop tensors where current variable is not the lowest in order
        bucket = []
        for key, tensor in candidate_tensors.items():
            sorted_tensor_indices = list(
                sorted(tensor.indices, key=int))
            if int(sorted_tensor_indices[0]) == variable:
                bucket.append(tensor)
        buckets.append(bucket)

    return buckets


def reorder_buckets(old_buckets, permutation):
    """
    Transforms bucket list according to the new order given by
    permutation. The variables are renamed and buckets are reordered
    to hold only gates acting on variables with strongly increasing
    index.

    Parameters
    ----------
    old_buckets : list of lists
          old buckets
    permutation : list
          permutation of variables

    Returns
    -------
    new_buckets : list of lists
          buckets reordered according to permutation
    label_dict : dict
          dictionary of new variable objects
          (as IDs of variables have been changed after reordering)
          in the form {old: new}
    """
    # import pdb
    # pdb.set_trace()
    if len(old_buckets) != len(permutation):
        raise ValueError('Wrong permutation: len(permutation)'
                         ' != len(buckets)')
    perm_dict = {}
    for n, idx in enumerate(permutation):
        if idx.name.startswith('v'):
            perm_dict[idx] = idx.copy(n)
        else:
            perm_dict[idx] = idx.copy(n, name=idx.name)

    n_variables = len(old_buckets)
    new_buckets = []
    for ii in range(n_variables):
        new_buckets.append([])

    for bucket in old_buckets:
        for tensor in bucket:
            new_indices = [perm_dict[idx] for idx in tensor.indices]
            bucket_idx = sorted(
                new_indices, key=int)[0].identity
            # we leave the variables permuted, as the permutation
            # information has to be preserved
            new_buckets[bucket_idx].append(
                tensor.copy(indices=new_indices)
            )

    return new_buckets, perm_dict


def test_bucket_graph_conversion(filename):
    """
    Test the conversion between Buckets and the contraction multigraph
    """
    import qtree.graph_model as gm

    # load circuit
    n_qubits, circuit = ops.read_circuit_file(filename)
    buckets, data_dict, bra_vars, ket_vars = circ2buckets(
        n_qubits, circuit)
    graph, *_ = gm.importers.circ2graph(
        n_qubits, circuit, omit_terminals=False)

    graph_from_buckets = gm.importers.buckets2graph(buckets)
    buckets_from_graph = graph2buckets(graph)

    buckets_equal = True
    for b1, b2 in zip(buckets, buckets_from_graph):
        if sorted(b1) != sorted(b2):
            buckets_equal = False
            break

    print('C->B, C->G->B: Buckets equal? : {}'.format(buckets_equal))
    print('C->G, C->B->G: Graphs equal? : {}'.format(
        nx.is_isomorphic(graph, graph_from_buckets)))


if __name__ == '__main__':
    test_bucket_graph_conversion('inst_2x2_7_0.txt')
