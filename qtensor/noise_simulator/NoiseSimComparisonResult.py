from helper_functions import *

from datetime import datetime
import time
import json
import jsbeautifier
import jsonpickle

from qtensor import QiskitQAOAComposer
import qiskit.providers.aer.noise as noise

from NoiseModel import NoiseModel

class NoiseSimComparisonResult:
    def __init__(self, qiskit_circ, qtensor_circ, qiskit_noise_model, qtensor_noise_model, n, p, d, name = ""):
        self.name = name
        self.experiment_date =  datetime.now().isoformat()
        self.experiment_start_time = time.time_ns() / (10 ** 9) # convert to floating-point seconds

        self.qiskit_circuit: QiskitQAOAComposer.circuit
        self.qtensor_circuit: list
        self.n: int 
        self.p: int
        self.d: int
        self.qiskit_noise_model: noise.NoiseModel
        self.qtensor_noise_model: NoiseModel

        self.qiskit_circuit = qiskit_circ
        self.qtensor_circuit = qtensor_circ
        self.n = n
        self.p = p
        self.d = d
        self.qiskit_noise_model = qiskit_noise_model
        self.qtensor_noise_model = qtensor_noise_model
        self.data = {}
        self._get_noise_model_string()
        self.pickled_qtensor_noise_model = jsonpickle.encode(self.qtensor_noise_model)

    def save_results_density(self, qiskit_density_matrix, qtensor_density_matrix, num_circs, num_circs_simulated, G, gamma, beta, qtensor_time_total, qiskit_time_total):
        self.qiskit_density_matrix = qiskit_density_matrix
        self.qtensor_density_matrix = qtensor_density_matrix
        self.num_circs = num_circs
        self.num_circs_simulated = num_circs_simulated
        self.graph = G
        self.gamma = gamma
        self.beta = beta
        self.qtensor_time_taken = qtensor_time_total
        self.qiskit_time_taken = qiskit_time_total
        self.experiment_end_time =  time.time_ns() / (10 ** 9) 
        self.total_time_taken = self.experiment_end_time - self.experiment_start_time
        self._calc_similarity()

    def save_result(self, qiskit_probs, qtensor_probs, exact_qtensor_amps, num_circs, num_circs_simulated, G, gamma, beta, qtensor_time_total, qiskit_time_total):
        self.qiskit_probs = qiskit_probs
        self.qtensor_probs = qtensor_probs
        self.num_circs = num_circs
        self.num_circs_simulated = num_circs_simulated
        self.graph = G
        self.gamma = gamma
        self.beta = beta
        self.qtensor_time_taken = qtensor_time_total
        self.qiskit_time_taken = qiskit_time_total
        self.experiment_end_time =  time.time_ns() / (10 ** 9) 
        self.experiment_time_taken = self.experiment_end_time - self.experiment_start_time
        self.exact_qtensor_amps = exact_qtensor_amps
        self._calc_similarity_of_probs()
        self._to_dict()

    def print_result(self):
        print("\nExperiment with n = {}, p = {}, d = {}, num_circs = {}, actual_num_circs = {}".format(self.n, self.p, self.d, self.num_circs, self.num_circs_simulated))
        # if type(self.fidelity) is str:
        #     print(
        #         f"{'Cosine Similarity:':<20}{np.round(self.cos_sim.real, 7):<10}",
        #         f"{'Total time taken:':<20}{self.experiment_time_taken:<10}")
        # else:
        print(
                f"{'Cosine Similarity:':<20}{np.round(self.cos_sim.real, 7):<10}",
                f"\n{'Noisy Fidelity:':<20}{np.round(self.noisy_fidelity.real, 7):<10}",
                f"\n{'Noiseless Fidelity:':<20}{np.round(self.noiseless_fidelity.real, 7):<10}",
                f"\n{'Uniform Qiskit Fidelity: ':<20}{np.round(self.uniform_qiskit_fidelity.real, 7):<10}",
                f"\n{'Uniform QTensor Fidelity: ':<20}{np.round(self.uniform_qtensor_fidelity.real, 7):<10}",
                f"\n{'Total time taken:':<20}{self.experiment_time_taken:<10}")

    def print_noise_model(self):
        print(self.noise_model_str)

    def save_experiment_to_file(self, outfile):
        options = jsbeautifier.default_options()
        options.indent_size = 2
        with open(outfile, 'a') as f:
            f.write(jsbeautifier.beautify(json.dumps(self.experiment_dict), options))
            f.write(', ')

    def _calc_similarity(self):
        qiskit_probs_root = np.sqrt(np.diagonal(self.qiskit_density_matrix))
        qtensor_probs_root = np.sqrt(np.diagonal(self.qtensor_density_matrix))
        
        self.cos_sim = cosine_similarity(qiskit_probs_root, qtensor_probs_root)
        self.fidelity = fidelity(self.qiskit_density_matrix.data, self.qtensor_density_matrix)

    def _calc_similarity_of_probs(self):
        qiskit_probs_root = np.sqrt(self.qiskit_probs)
        qtensor_probs_root = np.sqrt(self.qtensor_probs)
        noiseless_qtensor_probs_root = np.sqrt(np.abs(self.exact_qtensor_amps)**2)
        uniform_probs_root = np.sqrt(np.ones(2**self.n)/2**self.n)

        # we don't need to take the conjugate, as both population density vectors are strictly 
        # real by the time they have made it here. 

        """Measures the fidelity between the qiskit density matrix noisy state and the qtensoir stochastic noisy state"""
        self.noisy_fidelity = np.abs((np.inner(qiskit_probs_root, qtensor_probs_root)))**2

        """Measures the fidelity between a noisy qtensor state and the noiseless version of the same state"""
        self.noiseless_fidelity = np.abs((np.inner(noiseless_qtensor_probs_root, qtensor_probs_root)))**2

        """Measures the fidelity between a qiskit noisy state and a uniform distribution state"""
        self.uniform_qiskit_fidelity = np.abs((np.inner(uniform_probs_root, qiskit_probs_root)))**2

        """Measures the fidelity between a qtensor noisy state and a uniform distribution state"""
        self.uniform_qtensor_fidelity = np.abs((np.inner(uniform_probs_root, qtensor_probs_root)))**2
        self.cos_sim = cosine_similarity(self.qiskit_probs, self.qtensor_probs)

    def _to_dict(self):
        #self.experiment_dict['name'] = self.name
        self.data['num_qubits'] = self.n
        self.data['depth'] = self.p
        self.data['degree'] = self.d
        self.data['num_circs'] = self.num_circs
        self.data['actual_num_circs_simulated'] = self.num_circs_simulated
        self.data['gamma'] = self.gamma
        self.data['beta'] = self.beta
        self.data['cosine_similarity'] = self.cos_sim.real
        self.data['noisy_fidelity'] = self.noisy_fidelity
        self.data['noiseless_fidelity'] = self.noiseless_fidelity
        self.data['uniform_qiskit_fidelity'] = self.uniform_qiskit_fidelity
        self.data['uniform_qtensor_fidelity'] = self.uniform_qtensor_fidelity
        self.data['noise_model_string'] = self.noise_model_str_condensed
        self.data['noise_model_pickle'] = self.pickled_qtensor_noise_model
        self.data['experiment_date'] = self.experiment_date
        self.data['qtensor_time_taken'] = self.qtensor_time_taken
        self.data['qiskit_time_taken'] = self.qiskit_time_taken
        self.data['total_time_taken'] = self.experiment_time_taken

    def _get_noise_model_string(self):
        gates = self.qtensor_noise_model.noise_gates
        noise_model_str = 'Noise Model:'
        noise_model_str_condensed = ''
        for gate in gates:
            num_channels = len(self.qtensor_noise_model.noise_gates[gate].channels)
            if num_channels == 1:
                noise_model_str += '\nThe gate {} has 1 channel:\n'.format(gate)
                noise_model_str_condensed += "Gate: {}, Channel: ".format(gate)
            else:
                noise_model_str += '\nThe gate {} has {} channels:\n'.format(gate, num_channels)
                noise_model_str_condensed += "Gate: {}, {} Channels: ".format(gate, num_channels)
            all_channels_info = ''
            all_channels_info_short = ''
            for i in range(num_channels):
                name = self.qtensor_noise_model.noise_gates[gate].channels[i].name
                param = self.qtensor_noise_model.noise_gates[gate].channels[i].param 
                channel_info = 'Channel name: {}, with parameter: {}.\n'.format(name, param)
                channel_info_short = 'name: {}, param: {}. '.format(name, param)
                all_channels_info += channel_info
                all_channels_info_short += channel_info_short
            noise_model_str += all_channels_info
            noise_model_str_condensed += all_channels_info_short
        self.noise_model_str = noise_model_str
        self.noise_model_str_condensed = noise_model_str_condensed