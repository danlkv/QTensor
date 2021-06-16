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
    print('qcorig\n', circ)
    print('qc\n', qc)
    # note: .draw() returns a ``Drawing`` object, not string
    assert str(circ.draw()) == str(qc.draw())


if __name__=='__main__':
    test_builder()