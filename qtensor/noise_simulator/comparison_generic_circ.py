from generate_cz_rnd import get_cz_circ, CZBrickworkComposer, QiskitCZBrickworkComposer
import numpy as np
from qtree.operators import from_qiskit_circuit
from qiskit import execute, Aer
from qtensor import QiskitQAOAComposer
import qtensor
import networkx as nx
import qiskit.providers.aer.noise as noise
from qiskit.providers.aer import AerSimulator
import time

def get_circs(S, d):
    G = nx.random_regular_graph(3, S**2)

    comp = QiskitCZBrickworkComposer(S)
    comp.two_qubit_rnd(layers=d)

    ##
    gammabeta = np.array(qtensor.tools.BETHE_QAOA_VALUES[str(d)]['angles'])

    gamma = -gammabeta[:d]
    beta = gammabeta[d:]
    #comp = QiskitQAOAComposer(G, gamma=gamma, beta=beta)
    #comp.ansatz_state()
    n, qtensor_circ = from_qiskit_circuit(comp.circuit)

    comp.circuit = comp.circuit.reverse_bits()
    comp.circuit.save_density_matrix()
    comp.circuit.measure_all(add_bits = False)
    return qtensor_circ, comp.circuit

def simulate_qiskit_density_matrix(circuit, noise_model_qiskit, take_trace = True):
    start = time.time_ns() / (10 ** 9)
    backend = AerSimulator(method='density_matrix', noise_model=noise_model_qiskit)

    result = execute(circuit, backend, shots=1).result()
    #result = execute(circuit, Aer.get_backend('aer_simulator_density_matrix'), shots=1, noise_model=noise_model_qiskit).result()
    if take_trace:
        qiskit_probs = np.diagonal(result.results[0].data.density_matrix.real)
        end = time.time_ns() / (10 ** 9)
        qiskit_sim_time = end - start
        return qiskit_probs, qiskit_sim_time
    else:
        qiskit_density_matrix = result.results[0].data.density_matrix.real
        end = time.time_ns() / (10 ** 9)
        qiskit_sim_time = end - start
        return qiskit_density_matrix, qiskit_sim_time


if __name__=="__main__":
    from NoiseChannels import DepolarizingChannel
    from NoiseModel import NoiseModel
    from NoiseSimulator import NoiseSimulator

    prob_1 = 0.01
    prob_2 = 0.1

    S = 2
    d = 1
    qtensor_circ, qiskit_circ = get_circs(S, d)
    num_qubits = S**2
    # Qiskit Noise Model
    depol_chan_qiskit_1Q = noise.depolarizing_error(prob_1, 1)
    depol_chan_qiskit_2Q = noise.depolarizing_error(prob_2, 2)

    noise_model_qiskit = noise.NoiseModel()
    noise_model_qiskit.add_all_qubit_quantum_error(depol_chan_qiskit_1Q, ['x', 'y', 'z'])
    noise_model_qiskit.add_all_qubit_quantum_error(depol_chan_qiskit_1Q, ['rz', 'ry', 'rx', 'h'])
    noise_model_qiskit.add_all_qubit_quantum_error(depol_chan_qiskit_2Q, ['cx', 'cz'])

    # QTensor Noise Model
    depol_chan_qtensor_1Q = DepolarizingChannel(prob_1, 1)
    depol_chan_qtensor_2Q = DepolarizingChannel(prob_2, 2)

    noise_model_qtensor = NoiseModel()
    noise_model_qtensor.add_channel_to_all_qubits(depol_chan_qtensor_1Q, ['X', 'Y', 'Z'])
    noise_model_qtensor.add_channel_to_all_qubits(depol_chan_qtensor_1Q, ['XPhase', 'YPhase', 'ZPhase', 'H'])
    noise_model_qtensor.add_channel_to_all_qubits(depol_chan_qtensor_2Q, ['cX', 'cZ'])

    noise_sim = NoiseSimulator(noise_model_qtensor)


    for num_circs in [100, 300, 1000, 3000]:
        start = time.time_ns() / (10 ** 9)
        noise_sim.simulate_batch_ensemble(sum(qtensor_circ, []), num_circs, num_qubits)
        qtensor_probs = noise_sim.normalized_ensemble_probs 
        end = time.time_ns() / (10 ** 9)

        qiskit_probs, qiskit_time = simulate_qiskit_density_matrix(qiskit_circ, noise_model_qiskit, take_trace=True)
        #fidelity = np.sum(np.square(np.sqrt(qtensor_probs) * np.sqrt(qiskit_probs)))
        a = np.sqrt(qtensor_probs)
        b = np.sqrt(qiskit_probs)
        cos_sim = np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b)
        print("probs norm", print(sum(qtensor_probs)), print(sum(qiskit_probs)))
        fidelity = np.dot(a, b)**2
        print(f"num_circs: {num_circs}, Fidelity: {fidelity}, Cossim: {cos_sim}, TimeQiskit: {qiskit_time}, TimeQTensor: {end - start}")
