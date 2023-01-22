from qtensor.Simulate import QtreeSimulator, NumpyBackend
import qtree
from NoiseModel import NoiseModel

import numpy as np
import time

class NoiseSimulator(QtreeSimulator):
    def __init__(self, noise_model, bucket_backend=NumpyBackend(), optimizer=None, max_tw=None):
        super().__init__(bucket_backend, optimizer, max_tw)
        if not isinstance(noise_model, NoiseModel):
            raise ValueError("Error: noise_model value must be of type NoiseModel")
        self.noise_model = noise_model


    def simulate_batch_ensemble(self, qc, num_circs, batch_vars=0, peo=None):
        """Stoachastic noise simulator using statevector approach
        
        This method uses significantly less memory than the density matrix approach (2^n vs 4^n). However, one should note that 
        while this uses the statevector approach to simulate, the user never gets the amplitudes, only probabilities. 
        This is because we must take the modulus squared of each statevector in the ensemble to conserve probability. Thus the user will 
        receive a probability density vector, not a probability amplitude vector.  
        """
        start = time.time_ns() / (10 ** 9)
        if num_circs < 0 or not isinstance(num_circs, int):
            raise Exception("Error: The argument num_circs must be a positive integer")
        
        unnormalized_ensemble_probs = [0] * 2**batch_vars
        for _ in range(num_circs):
            noisy_state_amps = self.simulate_batch(qc, batch_vars, peo)
            noisy_state_probs = np.square(np.absolute(noisy_state_amps))
            unnormalized_ensemble_probs += noisy_state_probs
        
        normalized_ensemble_probs = unnormalized_ensemble_probs / num_circs
        self.normalized_ensemble_probs = normalized_ensemble_probs
        end = time.time_ns() / (10 ** 9)
        self.time_taken = end - start
        #return normalized_ensemble_probs

    def simulate_batch_ensemble_density(self, qc, num_circs, batch_vars=0, peo=None):
        """Stochastic noise simulator using density matrix apporach.
        
        Uses significantly more memory than the statevector approach, but gives more information. 
        The user will get the relative phaseses as well as the probabilities.
        """
        start = time.time_ns() / (10 ** 9)
        if num_circs < 0 or not isinstance(num_circs, int):
            raise Exception("Error: The argument num_circs must be a positive integer")
        
        unnormalized_ensemble_density_matrix = np.zeros(shape=(2**batch_vars, 2**batch_vars), dtype=complex)
        for _ in range(num_circs):
            noisy_state_amps = self.simulate_batch(qc, batch_vars, peo)
            conj_noisy_state_amps = noisy_state_amps.conj()
            noisy_state_density_matrix = np.outer(conj_noisy_state_amps, noisy_state_amps)
            unnormalized_ensemble_density_matrix += noisy_state_density_matrix
        
        self.normalized_ensemble_density_matrix = np.divide(unnormalized_ensemble_density_matrix, num_circs)
        end = time.time_ns() / (10 ** 9)
        self.time_taken = end - start
        #normalized_ensemble_density_matrix = np.divide(unnormalized_ensemble_density_matrix, num_circs)
        #return normalized_ensemble_density_matrix
    
    def simulate_ensemble(self, qc, num_circs):
        """Simulates and returns only the first amplitude of the ensemble"""
        return self.simulate_state(qc, num_circs)

    def simulate_state_ensemble(self, qc, num_circs, peo=None):
        return self.simulate_batch(qc, num_circs, peo=peo, batch_vars=0)

    #-- Internal helpers
    def _new_circuit(self, ideal_qc):
        self.all_gates = []
        for gate in ideal_qc:
            self.all_gates.append(gate)
            if gate.name in self.noise_model.noise_gates:
                self._apply_channel(gate)
        
    def _apply_channel(self, gate):
        """A noisy gate has all of the proper noise channels applied to it"""
        
        for i in range(len(self.noise_model.noise_gates[gate.name].channels)):
            error_name = self.noise_model.noise_gates[gate.name].channels[i].name
            if error_name == 'depolarizing':
                # The first tuple in the list of Kraus operators is always  II....I, n times, where n is the number of qubits
                # kraus_ops[0][1] refers to the second element of the first tuple in the list of kraus operators. 
                # The second element of the tuple contains the probability of that operator 
                prob_no_noise = self.noise_model.noise_gates[gate.name].channels[i].kraus_ops[0][1]
                p = np.random.uniform()
                if p >= prob_no_noise:
                    # In a depolarizing channel, the probability for any non II...I Kraus operator is always the same, 
                    # so we can just randomly choose any of the Kraus operators besides the first one in our noise_info list 
                    randIndex = np.random.randint(1, len(self.noise_model.noise_gates[gate.name].channels[i].kraus_ops))
                    # Get string of pauli(s), turn them into an iterable object, then add each pauli one by one to the circuit
                    kraus_op_name = self.noise_model.noise_gates[gate.name].channels[i].kraus_ops[randIndex][0]
                    paulis = iter(kraus_op_name)
                    for qubit in gate.qubits:
                        pauli = next(paulis)
                        if pauli != 'I':
                            self.all_gates.append(getattr(qtree.operators, pauli)(qubit))
                # else: p < prob_no_noise, so no noise is applied 
            
            elif error_name == 'bit_flip':
                if len(gate.qubits) > 1:
                    raise ValueError("Bit flip noise can only be applied to single qubit gates")

                prob_no_noise = self.noise_model.noise_gates[gate.name].channels[i].kraus_ops[0][1]
                p = np.random.uniform()
                if p >= prob_no_noise: 
                    self.all_gates.append(qtree.operators.X(gate.qubits[0]))
            
            elif error_name == 'phase_flip':
                if len(gate.qubits) > 1:
                    raise ValueError("Phase flip noise can only be applied to single qubit gates")

                prob_no_noise = self.noise_model.noise_gates[gate.name].channels[i].kraus_ops[0][1]
                p = np.random.uniform()
                if p >= prob_no_noise:
                    self.all_gates.append(qtree.operators.Z(gate.qubits[0]))








