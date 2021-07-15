import pytest
import qtensor
qic = qtensor.tools.qiskit_circuit

def test_name_to_gate_dict():
    op = qtensor.OpFactory.QtreeFactory
    name2gate = qic.get_name_to_gate_dict(op)
    assert name2gate['cz'] == op.cZ
    assert name2gate['z'] == op.Z

    op = qtensor.OpFactory.TorchFactory
    name2gate = qic.get_name_to_gate_dict(op)
    with pytest.raises(KeyError):
        assert name2gate['u1']

def test_builder():
    import qiskit
    from qiskit.circuit.library import TwoLocal
    N = 5

    builder = qtensor.QtreeBuilder(N)
    circ = TwoLocal(N, 'ry', 'cz', 'full', skip_final_rotation_layer=True, reps=3)
    circ = circ.assign_parameters([0.6]*len(circ.parameters))
    qic.build_from_qiskit(builder, circ)
    qc = builder.circuit
    print(qc)
    assert isinstance(qc[0], builder.operators.YPhase)

    import torch
    builder = qtensor.TorchBuilder(N)
    circ = TwoLocal(N, 'ry', 'cz', 'full', skip_final_rotation_layer=True, reps=3)
    for p in circ.parameters:
        p.value = torch.tensor(0.6)
    qic.build_from_qiskit(builder, circ)
    qc = builder.circuit
    print(qc)
    assert isinstance(qc[0], builder.operators.YPhase)


def compare_circuits(qc1, qc2):
    from qiskit.compiler import transpile
    basis_gates = [ "id", "x", "y", "z", "h", "s", "t", "sdg", "tdg", "rx", "ry", "rz", "rxx", "ryy", "cx", "cy", "cz", "ch", "crx", "cry", "crz", "swap", "cswap", "ccx", "cu1", "cu3", "u1", "u2", "u3", ]

    # One would think that comparing two objecs in Python should be as simple as
    # `a==b`, since every developer knows that if you override __eq__ method 
    # you should write something that works well

    # Not in qiskit

    # The __eq__ method uses circuit_to_dag, but circuit_to_dag doesn't correctly
    # handle the TwoLocal circuit, which is displayed as one giant gate on all
    # qubits. str(circuit) returns different stuff. However, the drawing script
    # handles it well, but you have to transpile the circuit to a basis set
    # Comparing unitaries is not an option here since the circuit has parameters
    # without values.

    def circ2id(qc):
        return (
            str(transpile(qc, basis_gates=basis_gates, optimization_level=0)
            .draw(output='text', fold=-1))
        )

    assert circ2id(qc1)==circ2id(qc2)


def test_qiskit_to_qiskit():
    import qiskit
    from qiskit.circuit.library import TwoLocal
    N = 5

    builder = qtensor.QiskitBuilder(N)
    circ = TwoLocal(N, 'ry', 'cz', 'full', skip_final_rotation_layer=True, reps=3)
    from qiskit.circuit import ParameterVector
    parameters = ParameterVector('theta', N * (3))
    param_iter = iter(parameters)

    import numpy as np
    for p in parameters:
        p.value = next(param_iter)
    import sympy
    circ = circ.assign_parameters(parameters)
    qic.build_from_qiskit(builder, circ)
    circ = TwoLocal(N, 'ry', 'cz', 'full', skip_final_rotation_layer=True, reps=3)
    circ = circ.assign_parameters([x*np.pi for x in parameters])
    circ.add_register(qiskit.ClassicalRegister(N,name='c'))
    qc = builder.circuit
    print('qcorig\n', str(circ.draw(fold=-1, output='text')))
    print('qcorig2\n', circ)
    print('qc\n', str(qc.draw(fold=-1, output='text')))
    compare_circuits(circ, qc)


if __name__=='__main__':
    test_builder()
