
QTENSOR_2_QISKIT_NAMES = {
    'XPhase': 'rx',
    'YPhase': 'ry',
    'ZPhase': 'rz',
    'I': 'id',
    'u1': 'u1',
    'u2': 'u2',
    'u3': 'u3',
    'X': 'x',
    'Y': 'y',
    'Z': 'z',
    'H': 'h',
    'cX': 'cx',
    'cY': 'cy',
    'cZ': 'cz',
}

def get_name_to_gate_dict(ops):
    gate_names = set.intersection(
        set(ops.__dict__.keys()), set(QTENSOR_2_QISKIT_NAMES.keys())
    )

    name_to_gate_dict = {v: ops.__dict__[k] for k, v in QTENSOR_2_QISKIT_NAMES.items()
                        if k in gate_names
                        }
    return name_to_gate_dict

def build_from_qiskit(builder, qiskit_circuit):
    circuit_qubits = qiskit_circuit.qubits

    output_circuit = []
    indexed_circuit = []

    # creates the indexed circuit, which replaces
    # qubits with their indices
    for circuit_operation in qiskit_circuit:
        qubit_list = []
        for circuit_operation_qubits in circuit_operation[1]:
            qubit_list.append(circuit_qubits.index(circuit_operation_qubits))

        indexed_circuit.append((circuit_operation[0], qubit_list))

    # for each operation, append the resulting internal gate representation(s)
    for circuit_operation in indexed_circuit:
        operations = _apply_qiskit_op(builder, circuit_operation)
        output_circuit.append(operations)

    # flatten the list. It may be of arbitrary depth due to nested gate definitions
    output_circuit = list(_flatten(output_circuit))

    return_circuit = []
    for gate in output_circuit:
        return_circuit.append([gate])

    qubit_count = len(qiskit_circuit.qubits)
    return qubit_count, return_circuit


def _apply_qiskit_op(builder, circuit_operation):
    """
    Read a modified circuit operation and return
    an internal representation of it

    Parameters
    ----------
    circuit_operation : (qiskit_instruction, [int])
            tuple of a qiskit instruction and the
            indices of the qubits it acts on

    Returns
    -------
    output_circuit : list of gate operations in
            internal representation
    """
    from qiskit import circuit
    ops = builder.operators

    name_to_gate_dict = get_name_to_gate_dict(ops)

    instruction = circuit_operation[0]
    instruction_qubits = circuit_operation[1]

    def unpack_parameter(x):
        if isinstance(x, circuit.Parameter):
            # a hack over qiskit's restrictive parameter assignment,
            # used to assign to tensors
            if not hasattr(x, 'value'):
                raise Exception('Parameter does not have value. assign .value to it')
            return x.value
        elif isinstance(x, circuit.ParameterExpression):
            return float(x._symbol_expr.evalf())
        else:
            return x
    # if the operation is part of the qiskit standard gate set
    if (instruction.name in name_to_gate_dict) and (isinstance(instruction, circuit.gate.Gate)):
        op_cls = name_to_gate_dict[instruction.name]
        if instruction.params:
            pex2num = lambda x: float(x._symbol_expr.evalf())
            params = [unpack_parameter(x) for x in instruction.params ]
            builder.apply_gate(op_cls, *instruction_qubits, *params)
        else:
            builder.apply_gate(op_cls, *instruction_qubits)


    # if the operation is a custom-defined gate whose basic gate definition needs to be found
    else:
        if (isinstance(instruction, circuit.Instruction)) and instruction.definition:
            for definition in instruction.definition:
                # replaces the definition of which qubits the custom gate acts on with the actual qubit indices
                new_qubit_list = []
                for definition_qubits in definition[1]:
                    new_qubit_list.append(instruction_qubits[definition_qubits.index])

                defined_operation = (definition[0], new_qubit_list)

                # recursively call the function until all nested custom gates are defined
                _apply_qiskit_op(builder, defined_operation)



def _flatten(l):
    """
    https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    Parameters
    ----------
    l: iterable
        arbitrarily nested list of lists

    Returns
    -------
    generator of a flattened list
    """
    import collections
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from _flatten(el)
        else:
            yield el
