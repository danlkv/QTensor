import qtensor as qtn
import qtree
import numpy as np
import qiskit

from qiskit.circuit import QuantumCircuit
from qiskit import execute
from qiskit import Aer
from qiskit.optimization.ising import max_cut, tsp,common
import cirq

def test_warm_simple():
    angle = 0.523
    circ = []
    rotated =[[4.872435284588963, 3.140924329249939],
     [6.2089546789130585, 0.0],
     [0.44075316687891064, 1.0512066194209708],
     [3.581626554108839, 1.046481705810185],
     [3.580582956190857, 2.0938670181912022],
              [0.441558317709732, 2.098012455304285]][:4]
    N = len(rotated)
    for qubit, (z, x) in enumerate(rotated):
        circ.append(qtree.operators.rx([x], qubit))
        circ.append(qtree.operators.rz([z+np.pi/2], qubit))
    sim = qtn.QtreeSimulator()
    res_qtn = sim.simulate_batch(circ, batch_vars=N)

    cirq1=cirq.Circuit()
    qubits = cirq.LineQubit.range(N)


    builder = qtn.CirqBuilder(N)
    for qubit, (z, x) in zip(qubits, rotated):
        cirq1.append(cirq.rx(x).on(qubit))
        cirq1.append(cirq.rz(z+np.pi/2).on(qubit))
    print(circ)
    print(cirq1)
    #cirq1.append(cirq.rz(angle).on(qubits[0]))
    s=cirq.Simulator()
    sim=s.simulate(cirq1)
    res_cq = sim.final_state_vector
    print(res_qtn)
    print(res_cq)
    print(res_cq/res_qtn)
    assert False



def test_warm_start():
    rotated =[[4.872435284588963, 3.140924329249939],
     [6.2089546789130585, 0.0],
     [0.44075316687891064, 1.0512066194209708],
     [3.581626554108839, 1.046481705810185],
     [3.580582956190857, 2.0938670181912022],
     [0.441558317709732, 2.098012455304285]]
    rotated = rotated[:]
    N = len(rotated)
    G = qtn.toolbox.random_graph(degree=3, nodes=N, seed=10)
    qt_sim=qtn.QAOASimulator.WarmStartQAOASimulator(qtn.WarmStartQtreeQAOAComposer, solution=rotated)
    qtn_res = qt_sim.energy_expectation(G, gamma=[0], beta=[0])[0]

    ##Cirq construction
    X_rotations = [ rotated[i][1] for i in range(len(rotated)) ]
    Z_rotations = [ rotated[i][0] for i in range(len(rotated)) ]
    cirq1=cirq.Circuit()

    for (q,Xrot,Zrot) in zip(cirq.LineQubit.range(len(rotated)),X_rotations,Z_rotations): 
        cirq1.append(cirq.rx(Xrot).on(q))
        cirq1.append(cirq.rz(Zrot+np.pi/2).on(q))
    s=cirq.Simulator()
    sim=s.simulate(cirq1)

    ##Qiskit construction
    circuit = QuantumCircuit(G.number_of_nodes())
    for (q,Xrot,Zrot) in zip(range(G.number_of_nodes()-1,-1,-1),X_rotations,Z_rotations): 
        circuit.rx(Xrot,q)
        circuit.rz(Zrot+np.pi/2,q)
    ex1 = execute(circuit,backend=Aer.get_backend("statevector_simulator")).result()
    warm_state = ex1.get_statevector()
    w = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            temp = G.get_edge_data(i,j,default=0)
            if temp != 0:
                w[i,j] = 1#temp['weight'] 
    qubitOp, offset = max_cut.get_operator(w)
    qiskit_res = -(qubitOp.evaluate_with_statevector(warm_state)[0].real+offset)

    cirq_res = -(qubitOp.evaluate_with_statevector(sim.final_state_vector)[0].real+offset)
    comp = qtn.WarmStartQtreeQAOAComposer(G, gamma=[0], beta=[0], solution=rotated)
    comp.ansatz_state()
    print(comp.circuit)
    print('qtensor circuit', comp.circuit[:2*N])
    print('circq circuit', cirq1)
    state = qt_sim.simulate_batch(comp.circuit[:2*N], batch_vars=N)
    print('cirq state', sim.final_state_vector)
    print('qtensor state', state)
    print('compare', state/sim.final_state_vector)
    print('qiskit state', warm_state)
    print(warm_state/sim.final_state_vector)
    print(cirq_res)
    assert qtn_res==cirq_res
    assert qtn_res==qiskit_res

    qiskit_res = -(qubitOp.evaluate_with_statevector(warm_state)[0].real+offset)
    print(qiskit_res)

