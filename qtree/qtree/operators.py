"""
This module implements quantum gates from the CMON set of Google
"""
import numpy as np
import re
import cirq

from fractions import Fraction
from qtree.logger_setup import log

import qtree.system_defs as defs
import uuid


class Gate:
    """
    Base class for quantum gates.

    Attributes
    ----------
    name: str
            The name of the gate

    parameters: dict
             Parameters used by the gate (may be empty)

    qubits: tuple
            Qubits the gate acts on

    changed_qubits : tuple
            Tuple of ints which states what qubit's bases are changed
            (along which qubits the gate is not diagonal).

    cirq_op: Cirq.GridQubit
            Cirq 2D gate. Used for unit tests. Optional

    Methods
    -------
    gen_tensor(): numpy.array
            The gate tensor. For each qubit a gate
            either introduces a new variable (non-diagonal gate, like X)
            or does not (diagonal gate, like T). Multiqubit gates
            can be diagonal on some of the variables, and not diagonal on
            others (like ccX). The order of dimensions IS ALWAYS
            (new_a, a, b_new, b, c, d_new, d, ...)

    is_parametric(): bool
            Returns False for gates without parameters
    """

    def __init__(self, *qubits):
        self._qubits = tuple(qubits)
        # supposedly unique id for an instance
        self._parameters = {}
        self._check_qubit_count(qubits)

    def _check_qubit_count(self, qubits):
        n_qubits = len(self.gen_tensor().shape) - len(
            self._changes_qubits)
        # return back the saved version

        if len(qubits) != n_qubits:
            raise ValueError(
                "Wrong number of qubits for gate {}:\n"
                "{}, required: {}".format(
                    self.name, len(qubits), n_qubits))

    @property
    def name(self):
        return type(self).__name__

    def gen_tensor(self):
        raise NotImplementedError()

    @property
    def parameters(self):
        return self._parameters

    def is_parametric(self):
        return len(self.parameters) > 0

    @property
    def qubits(self):
        return self._qubits

    @property
    def changed_qubits(self):
        return tuple(self._qubits[idx] for idx in self._changes_qubits)

    def to_cirq_1d_circ_op(self):
        return self.cirq_op(
            *[cirq.LineQubit(qubit) for qubit in self._qubits]
        )

    def __str__(self):
        return ("{}".format(type(self).__name__) +
                "({})".format(','.join(map(str, self._qubits)))
        )

    def __repr__(self):
        return self.__str__()


class ParametricGate(Gate):
    """
    Gate with parameters.

    Attributes
    ----------
    name: str
            The name of the gate

    parameters: dict
             Parameters used by the gate (may be empty)

    qubits: tuple
            Qubits the gate acts on

    changed_qubits : tuple
            Tuple of ints which states what qubit's bases are changed
            (along which qubits the gate is not diagonal).

    cirq_op: Cirq.GridQubit
            Cirq 2D gate. Used for unit tests. Optional

    Methods
    -------
    gen_tensor(\\**parameters={}): numpy.array
            The gate tensor. For each qubit a gate
            either introduces a new variable (non-diagonal gate, like X)
            or does not (diagonal gate, like T). Multiqubit gates
            can be diagonal on some of the variables, and not diagonal on
            others (like ccX). The order of dimensions IS ALWAYS
            (new_a, a, b_new, b, c, d_new, d, ...)

    is_parametric(): bool
            Returns True
    """
    def __init__(self, *qubits, **parameters):
        self._qubits = tuple(qubits)
        # supposedly unique id for an instance
        self._parameters = parameters
        self._check_qubit_count(qubits)

    def _check_qubit_count(self, qubits):
        # fill parameters and save a copy
        filled_parameters = {}
        for par, value in self._parameters.items():
            if isinstance(value, placeholder):
                filled_parameters[par] = np.zeros(value.shape)
            else:
                filled_parameters[par] = value
        parameters = self._parameters

        # substitute parameters by filled parameters
        # to evaluate tensor shape
        self._parameters = filled_parameters
        n_qubits = len(self.gen_tensor().shape) - len(
            self._changes_qubits)
        # return back the saved version
        self._parameters = parameters

        if len(qubits) != n_qubits:
            raise ValueError(
                "Wrong number of qubits: {}, required: {}".format(
                    len(qubits), n_qubits))

    def gen_tensor(self, **parameters):
        if len(parameters) == 0:
            return self._gen_tensor(**self._parameters)
        else:
            return self._gen_tensor(**parameters)

    def __str__(self):
        par_str = (",".join("{}={}".format(
            param_name,
            '?.??' if isinstance(param_value, placeholder)
            else '{:.2f}'.format(float(param_value)))
                            for param_name, param_value in
                            sorted(self._parameters.items(),
                                   key=lambda pair: pair[0])))

        return ("{}".format(type(self).__name__) + "[" + par_str + "]" +
                "({})".format(','.join(map(str, self._qubits))))


def op_scale(factor, operator, name):
    """
    Scales a gate class by a scalar. The resulting class
    will have a scaled tensor

    It is not recommended to use this many times because of
    possibly low performance

    Parameters
    ----------
    factor: float
          scaling factor
    operator: class
          operator to modify
    name: str
          Name of the new class
    Returns
    -------
    class
    """
    def gen_tensor(self):
        return factor * operator.gen_tensor(operator)

    attr_dict = {attr: getattr(operator, attr) for attr in dir(operator)}
    attr_dict['gen_tensor'] = gen_tensor

    return type(name, (operator, ), attr_dict)


class M(Gate):
    """
    Measurement gate. This is essentially the identity operator, but
    it forces the inntroduction of a variable in the graphical model
    """
    def gen_tensor(self):
        return np.array([[1, 0], [0, 1]], dtype=defs.NP_ARRAY_TYPE)

    _changes_qubits = (0, )
    cirq_op = cirq.I


class I(Gate):
    def gen_tensor(self):
        return np.array([1, 1], dtype=defs.NP_ARRAY_TYPE)

    _changes_qubits = tuple()
    cirq_op = cirq.I


class H(Gate):
    """
    Hadamard gate
    """
    def gen_tensor(self):
        return 1/np.sqrt(2) * np.array([[1,  1],
                                        [1, -1]],
                                       dtype=defs.NP_ARRAY_TYPE)
    _changes_qubits = (0, )
    cirq_op = cirq.H


class cZ(Gate):
    """
    Controlled :math:`Z` gate
    """
    def gen_tensor(self):
        return np.array([[1, 1],
                         [1, -1]],
                        dtype=defs.NP_ARRAY_TYPE)
    _changes_qubits = tuple()
    cirq_op = cirq.CZ


class Z(Gate):
    """
    :math:`Z`-gate
    """
    def gen_tensor(self):
        return np.array([1, -1],
                        dtype=defs.NP_ARRAY_TYPE)

    _changes_qubits = tuple()
    cirq_op = cirq.Z


class T(Gate):
    """
    :math:`T`-gate
    """
    def gen_tensor(self):
        return np.array([1, np.exp(1.j*np.pi/4)],
                        dtype=defs.NP_ARRAY_TYPE)

    _changes_qubits = tuple()
    cirq_op = cirq.T


class X_1_2(Gate):
    """
    :math:`X^{1/2}`
    gate
    """
    def gen_tensor(self):
        return (Fraction(1, 2) *
                np.array([[1 + 1j, 1 - 1j],
                          [1 - 1j, 1 + 1j]])
        ).astype(defs.NP_ARRAY_TYPE)

    _changes_qubits = (0, )

    def cirq_op(self, x): return cirq.X(x)**0.5


class Y_1_2(Gate):
    r"""
    :math:`Y^{1/2}` gate
    """
    def gen_tensor(self):
        return (Fraction(1, 2) *
                np.array([[1 + 1j, -1 - 1j],
                          [1 + 1j, 1 + 1j]])
        ).astype(defs.NP_ARRAY_TYPE)

    _changes_qubits = (0, )

    def cirq_op(self, x): return cirq.Y(x)**0.5


class X(Gate):
    def gen_tensor(self):
        return np.array([[0, 1],
                         [1, 0]],
                        dtype=defs.NP_ARRAY_TYPE)

    _changes_qubits = (0, )

    def cirq_op(self, x): return cirq.X(x)


class cX(Gate):
    def gen_tensor(self):
        return np.array([[[1., 0.],
                          [0., 1.]],
                         [[0., 1.],
                          [1., 0.]]])

    _changes_qubits = (1, )
    cirq_op = cirq.CNOT


class Y(Gate):
    def gen_tensor(self):
        return np.array([[0, -1j],
                         [1j, 0]],
                        dtype=defs.NP_ARRAY_TYPE)

    _changes_qubits = (0, )

    def cirq_op(self, x): return cirq.Y(x)


class ZPhase(ParametricGate):
    """Arbitrary :math:`Z` rotation
    [[1, 0],
    [0, g]],  where

    g = exp(i·π·t)
    """

    _changes_qubits = tuple()

    @staticmethod
    def _gen_tensor(**parameters):
        """Rotation along Z axis"""
        alpha = parameters['alpha']
        return np.array([1., np.exp(1j * np.pi * alpha)])

    def cirq_op(self, x): return cirq.ZPowGate(
            exponent=float(self._parameters['alpha']))(x)


def rz(rads):
    """Arbitrary :math:`Z` rotation"""

    return ZPhase(alpha=rads/np.pi)


class XPhase(ParametricGate):
    """Arbitrary :math:`X` rotation
    [[g·c, -i·g·s],
    [-i·g·s, g·c]], where

    c = cos(π·alpha/2), s = sin(π·alpha/2), g = exp(i·π·alpha/2).
    """

    _changes_qubits = (0, )

    @staticmethod
    def _gen_tensor(**parameters):
        """Rotation along X axis"""
        alpha = parameters['alpha']

        c = np.cos(np.pi*alpha/2)
        s = np.sin(np.pi*alpha/2)
        g = np.exp(1j*np.pi*alpha/2)

        return np.array([[g*c, -1j*g*s],
                         [-1j*g*s, g*c]])

    def cirq_op(self, x): return cirq.XPowGate(
            exponent=float(self._parameters['alpha']))(x)


def rx(rads):
    """Arbitrary :math:`X` rotation"""

    return XPhase(alpha=rads/np.pi - 0.5)


class XPhase(ParametricGate):
    """Arbitrary :math:`X` rotation
    [[g·c, -g·s],
    [g·s, g·c]], where

    c = cos(π·alpha/2), s = sin(π·alpha/2), g = exp(i·π·alpha/2).
    """

    _changes_qubits = (0, )

    @staticmethod
    def _gen_tensor(**parameters):
        """Rotation along X axis"""
        alpha = parameters['alpha']

        c = np.cos(np.pi*alpha/2)
        s = np.sin(np.pi*alpha/2)
        g = np.exp(1j*np.pi*alpha/2)

        return np.array([[g*c, -1j*g*s],
                         [-1j*g*s, g*c]])

    def cirq_op(self, x): return cirq.XPowGate(
            exponent=float(self._parameters['alpha']))(x)


def read_circuit_file(filename, max_depth=None):
    """
    Read circuit file and return quantum circuit in the
    form of a list of lists

    Parameters
    ----------
    filename : str
             circuit file in the format of Sergio Boixo
    max_depth : int
             maximal depth of gates to read

    Returns
    -------
    qubit_count : int
            number of qubits in the circuit
    circuit : list of lists
            quantum circuit as a list of layers of gates
    """
    label_to_gate_dict = {
        'i': I,
        'h': H,
        't': T,
        'z': Z,
        'cz': cZ,
        'x': X,
        'y': Y,
        'x_1_2': X_1_2,
        'y_1_2': Y_1_2,
    }

    operation_search_patt = r'(?P<operation>' + r'|'.join(label_to_gate_dict.keys()) + r')(?P<qubits>( \d+)+)'

    log.info("reading file {}".format(filename))
    circuit = []
    circuit_layer = []

    with open(filename, "r") as fp:
        qubit_count = int(fp.readline())
        log.info("There are {:d} qubits in circuit".format(qubit_count))
        n_ignored_layers = 0
        current_layer = 0

        for idx, line in enumerate(fp):
            m = re.search(r'(?P<layer>[0-9]+) (?=[a-z])', line)
            if m is None:
                raise Exception("file format error at line {}".format(idx))
            # Read circuit layer by layer
            layer_num = int(m.group('layer'))

            if max_depth is not None and layer_num > max_depth:
                n_ignored_layers = layer_num - max_depth
                continue

            if layer_num > current_layer:
                circuit.append(circuit_layer)
                circuit_layer = []
                current_layer = layer_num

            op_str = line[m.end():]
            m = re.search(operation_search_patt, op_str)
            if m is None:
                raise Exception("file format error in {}".format(op_str))

            op_identif = m.group('operation')

            q_idx = tuple(int(qq) for qq in m.group('qubits').split())

            op = label_to_gate_dict[op_identif](*q_idx)
            circuit_layer.append(op)

        circuit.append(circuit_layer)  # last layer

        if n_ignored_layers > 0:
            log.info("Ignored {} layers".format(n_ignored_layers))

    return qubit_count, circuit


def read_qasm_file(filename, max_depth=None):
    """
    Read circuit file in the QASM format and return
    quantum circuit in the form of a list of lists

    Parameters
    ----------
    filename : str
             circuit file in the QASM format
    max_depth : int
             maximal depth of gates to read

    Returns
    -------
    qubit_count : int
            number of qubits in the circuit
    circuit : list of lists
            quantum circuit as a list of layers of gates
    """

    pass


class placeholder:
    """
    Class for placeholders. Placeholders are used to implement
    symbolic computation. This class is very similar to the
    Tensorflow's placeholder class.

    Attributes
    ----------
    name: str, default None
          Name of the placeholder (for clarity)
    shape: tuple, default None
          Shape of the tensor the placeholder represent
    """
    def __init__(self, name=None, shape=()):
        self._name = name
        self._shape = shape
        self._uuid = uuid.uuid4()

    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._shape

    @property
    def uuid(self):
        return self._uuid
