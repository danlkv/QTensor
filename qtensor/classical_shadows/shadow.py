from qtensor.Simulate import QtreeSimulator, NumpyBackend
from qtree import operators as op
from qtree.operators import Gate 
import numpy as np
from qtensor.noise_simulator.helper_functions import decimal_to_binary
import copy

class Classical_Shadow(QtreeSimulator): 
    def __init__(self, bucket_backend=NumpyBackend(), optimizer=None, max_tw=None):
        super().__init__(bucket_backend, optimizer, max_tw)

    def get_snapshot(self, circuit, num_snapshots, num_qubits):
        """
        Gets the classical shadows from a given circuit

        Args: 
            circuit (list): the circuit to be simulated
            num_snapshots (int): the number of shadows in the collection
            num_qubits (int): the number of qubits in the circuit

        Returns: 

        """
        obseravbles = [op.X, op.Y, op.Z]
        basis_measurements = [[op.H], [op.Sdag, op.H], [op.M]]
        observable_ids = np.random.randint(0, 3, size=(num_snapshots, num_qubits))
        # observable_ids = np.random.randint(1, 2, size=(num_snapshots, num_qubits))
        snapshots = np.zeros((num_snapshots, num_qubits))

        for snapshot in range(num_snapshots):
            # Generates a list of paulis applied to the i-th qubit, then appends the list to the circuit
            snapshot_circ = copy.deepcopy(circuit)
            # print((basis_measurements[int(observable_ids[snapshot, i])](i) forH  i in range(num_qubits)))
            for i in range(num_qubits):
                for clifford in basis_measurements[int(observable_ids[snapshot, i])]:
                    snapshot_circ.append(clifford(i))
            # snapshot_circ += (basis_measurements[int(observable_ids[snapshot, i])](i) for i in range(num_qubits))
            print(snapshot_circ)
            snapshots[snapshot, :] = self._get_measurement(snapshot_circ, num_qubits)
            # print(snapshot_circ)
        return (snapshots, observable_ids)

    def _get_measurement(self, circuit, num_qubits):
            """
            Simulates state after applying a pauli to each qubit. Measure the state to obtain a bitstring, and then map 0 -> 1 and 1 -> -1 

            Args: 
                circuit (list): the circuit to be simulated
                num_qubits (int): the number of qubits in the circuit
            
            Returns:
                eigenvalues: the bitstring obtained probabilisitically from the statevector, with its values mapped to 1 and -1
            """

            sim = QtreeSimulator()
            statevector = sim.simulate_batch(circuit, batch_vars = num_qubits)
            # print("statevector:", statevector)
            # print(statevector)
            probs = np.square(np.absolute(statevector))
            probs = [round(elem, 6) for elem in probs]
            #print("statevector:", probs)
            measurement = int(np.random.choice(np.arange(len(probs)), size = 1, p = probs))
            #print(measurement, type(measurement))
            length = int(np.ceil(np.log(2**num_qubits + 1)/np.log(2)) - 1)
            measurement = str(decimal_to_binary(measurement).zfill(length))
            #print(measurement, type(measurement))
            measurement = np.asarray([int(measurement[i]) for i in range(len(measurement))])
            #print(measurement, type(measurement))
            eigenvalues = np.piecewise(measurement, [measurement == 0, measurement == 1], [1, -1])
            # eigenvalues = []
            # for i in range(len(measurement)):
            #     eigenvalue = np.piecewise(measurement[i], [measurement[i] == '0', measurement[i] == '1'], [1, -1])
            #     print(f"eiegenvalue: {eigenvalue}")
            #     eigenvalues.append(eigenvalue)
            #     print(f"measurement[i]: {measurement[i]}, eigevnalues[i]: {eigenvalues[i]}")
            #eigenvalues = [np.piecewise(measurement[i], [measurement[i] == '0', measurement[i] == '1'], [1, -1]) for i in range(len(measurement))] 
            #print(eigenvalues)
            return eigenvalues
    
    def _snapshot_state(self, measurement_outcomes, obseravbles):
        num_qubits = len(measurement_outcomes)

        zero_state = np.array([[1, 0], [0, 0]])
        one_state = np.array([[0, 0], [0, 1]])

        # H = op.H(Gate).gen_tensor()
        H = 1/np.sqrt(2)*np.array([[1,1],[1,-1]])
        Sdag = np.array([[1,0],[0,-1j]],dtype=complex)
        I = np.identity(2)
        # print(H)
        # ZPhase = op.ZPhase(Gate).gen_tensor({'alpha': np.pi/2})
        # ZPhase = np.array([[1, 0], [0, -1j]], dtype=complex)
        # I = op.M(Gate).gen_tensor()
        # I = np.array([[1, 0], [0, 1]], dtype=complex)
        #unitaries = [H, H_Sdag(Gate).gen_tensor(), I]
        # Sdag = op.Sdag(Gate).gen_tensor()
        unitaries = [H, H @ Sdag , I]
        # unitaries = [op.X.gen_tensor(Gate), op.Y.gen_tensor(Gate), op.Z.gen_tensor(Gate)]


        rho_snapshot = [1]
        for i in range(num_qubits):
            state = zero_state if measurement_outcomes[i] == 1 else one_state
            U = unitaries[int(obseravbles[i])]

            local_rho = 3 * U.conj().T @ state @ U - I
            rho_snapshot = np.kron(rho_snapshot, local_rho)

        return rho_snapshot



    def get_approximate_state(self, shadow): 
        """
        Constructs and approximate state from the n snapshots in the shadow

        Args: 
            shadow (tuple): a classical shadow 
        """
        num_snapshots, num_qubits = shadow[0].shape
        measurement_outcomes, observables = shadow

        shadow_rho = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)

        for i in range(num_snapshots):
            shadow_rho += self._snapshot_state(measurement_outcomes[i], observables[i])

        return shadow_rho / num_snapshots
    

class H_Sdag(Gate):
    name = 'H_Sdag'
    _changes_qubits = tuple()
    # _changes_qubits = (0,)
    def gen_tensor(self):
        return op.H(Gate).gen_tensor() @ op.Sdag(Gate).gen_tensor() 
        # return 1/np.sqrt(2) * np.array([[1.+0.j, 0.-1j],
        #                                 [1.+0.j, 0.+1j]])
    
class Sdag_H(Gate):
    name = 'Sdag_H'
    _changes_qubits = tuple()
    # _changes_qubits = (0,)
    def gen_tensor(self):
        return op.Sdag(Gate).gen_tensor() @ op.H(Gate).gen_tensor() 
        # return 1/np.sqrt(2) * np.array([[1.+0.j, 1.+0j],
        #                                 [0.-1.j, 0.+1j]])

