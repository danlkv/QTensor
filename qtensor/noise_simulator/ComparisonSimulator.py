from NoiseChannels import *
from NoiseSimulator import * 
from NoiseModel import *
from NoiseSimComparisonResult import *
from helper_functions import *
from qiskit import execute, Aer
from qtensor.tests.test_composers import *
from qtensor import QiskitQAOAComposer
from qtree.operators import from_qiskit_circuit
from qtensor import tools
from itertools import repeat
from qiskit import execute, Aer
from qiskit.providers.aer import AerSimulator

class ComparisonSimulator:
    def __init__(self):
        pass

class QAOAComparisonSimulator(ComparisonSimulator):
    def __init__(self, n: int, p: int, d: int, noise_model_qiskit: noise.NoiseModel, noise_model_qtensor: NoiseModel, num_circs_list: list):
        self.n = n
        self.p = p
        self.d = d
        self.noise_model_qiskit = noise_model_qiskit
        self.noise_model_qtensor = noise_model_qtensor
        self.num_circs_list = num_circs_list
        self.num_circs_simulated = []
        self.results = []

        # We can save on simulation time by not recomputing previous work. This is done by 
        # using the results of a previous, smaller ensemble simulation: 
        # i.e. if we ran 1,000 circuits in the previous simulation, and now we want to run 10,000 
        # circuits using the exact same circuit and parameters, we actually only need to run
        # 9,000 circuits, add the previous results, and renormalize. 
        # The downside of this approach is that it requires the storage of an extra state 
        # in memory. Therefore it should only be done for simulations that do not require 
        # all of the available memory
        self.recompute_previous_ensemble: bool

        self._check_params()
        self.num_circs_list.sort()

    def qtensor_qiskit_noisy_qaoa(self, recompute_previous_ensemble: bool = False, print_stats: bool = False):
        self.recompute_previous_ensemble = recompute_previous_ensemble
        # Prepare circuits, simulator
        G, gamma, beta = get_qaoa_params(n = self.n, p = self.p, d = self.d)
        self._get_circuits(G, gamma, beta)
        noise_sim = NoiseSimulator(self.noise_model_qtensor)
        exact_sim = QtreeSimulator()

        # Run simulation
        for num_circs, i in zip(self.num_circs_list, range(len(self.num_circs_list))):
            result = NoiseSimComparisonResult(self.qiskit_circ, self.qtensor_circ, self.noise_model_qiskit, 
                self.noise_model_qtensor, self.n, self.p, self.d)

            if i == 0 or recompute_previous_ensemble == False:
                self.num_circs_simulated.append(num_circs)
                noise_sim.simulate_batch_ensemble(sum(self.qtensor_circ, []), num_circs, self.num_qubits)
                #print("num_circs: {}".format(num_circs))
                self.qtensor_probs = noise_sim.normalized_ensemble_probs
            else: 
                actual_num_circs = num_circs - self.num_circs_list[i - 1]
                self.num_circs_simulated.append(actual_num_circs)
                #print("num_circs: {}, actual_num_circs: {}".format(num_circs, actual_num_circs_to_sim))
                noise_sim.simulate_batch_ensemble(sum(self.qtensor_circ, []), actual_num_circs, self.num_qubits)
                prev_qtensor_probs = self.prev_probs
                curr_qtensor_probs = noise_sim.normalized_ensemble_probs
                self.qtensor_probs = (curr_qtensor_probs + prev_qtensor_probs) / 2
            
            if recompute_previous_ensemble == True:
                #self._check_correct_num_circs_simulated(i)
                self.prev_probs = self.qtensor_probs

            qtensor_sim_time = noise_sim.time_taken
            self.simulate_qiskit_density_matrix(self.qiskit_circ, self.noise_model_qiskit)    
            self.exact_qtensor_amps = exact_sim.simulate_batch(sum(self.qtensor_circ, []), batch_vars=self.num_qubits)

            # Save results
            result.save_result(self.qiskit_probs, self.qtensor_probs, self.exact_qtensor_amps, num_circs,
                self.num_circs_simulated[i], G, gamma, beta, qtensor_sim_time, self.qiskit_sim_time)
            self.results.append(result.data)
            if print_stats:
                result.print_result()

    def _mpi_parallel_unit(self, args):
        noise_sim, qtensor_circ, num_circs, num_qubits = args
        #print("this workers num_circs: ", num_circs)
        noise_sim.simulate_batch_ensemble(sum(qtensor_circ, []), num_circs, num_qubits)
        fraction_of_qtensor_probs = noise_sim.normalized_ensemble_probs
        return fraction_of_qtensor_probs

    def qtensor_qiskit_noisy_qaoa_mpi(self, num_nodes: int, num_jobs_per_node: int, recompute_previous_ensemble: bool = False, print_stats: bool = True, pbar: bool = True):
        self.num_nodes = num_nodes 
        self.num_jobs_per_node = num_jobs_per_node
        self.recompute_previous_ensemble = recompute_previous_ensemble

        # Prepare circuit, simulator, and area to save results, 
        G, gamma, beta = get_qaoa_params(n = self.n, p = self.p, d = self.d)
        self._get_circuits(G, gamma, beta)
        self.noise_sim = NoiseSimulator(self.noise_model_qtensor)
        exact_sim = QtreeSimulator()

        for num_circs, i in zip(self.num_circs_list, range(len(self.num_circs_list))):
            result = NoiseSimComparisonResult(self.qiskit_circ, self.qtensor_circ, self.noise_model_qiskit, 
                self.noise_model_qtensor, self.n, self.p, self.d)
            self._get_args(i)
            qtensor_probs_list = tools.mpi.mpi_map(self._mpi_parallel_unit, self._arggen, pbar=pbar, total=num_nodes)
            if qtensor_probs_list:
                if i == 0 or recompute_previous_ensemble == False: 
                    self.qtensor_probs = sum(qtensor_probs_list) / self._total_jobs
                else:
                    #prev_qtensor_probs = self.results[i - 1].qtensor_probs
                    prev_qtensor_probs = self.prev_probs
                    curr_qtensor_probs = sum(qtensor_probs_list) / self._total_jobs
                    self.qtensor_probs = (curr_qtensor_probs + prev_qtensor_probs) / 2
                qtensor_sim_time = self.noise_sim.time_taken
                if recompute_previous_ensemble == True:
                    self._check_correct_num_circs_simulated(i)
                    self.prev_probs = self.qtensor_probs

                self.simulate_qiskit_density_matrix(self.qiskit_circ, self.noise_model_qiskit)
                self.exact_qtensor_amps = exact_sim.simulate_batch(sum(self.qtensor_circ, []), batch_vars=self.num_qubits)
                # Save results 
                result.save_result(self.qiskit_probs, self.qtensor_probs, self.exact_qtensor_amps, num_circs,
                    self.num_circs_simulated[i], G, gamma, beta, qtensor_sim_time, self.qiskit_sim_time)
                self.results.append(result.data) 
                if print_stats:
                    tools.mpi.print_stats()
                    result.print_result()

    def qtensor_qiskit_noisy_qaoa_density(self, recompute_previous_ensemble: bool = False):
        self.recompute_previous_ensemble = recompute_previous_ensemble
        # Prepare circuits, simulator, and area to save results
        G, gamma, beta = get_qaoa_params(n = self.n, p = self.p, d = self.d)
        self._get_circuits(G, gamma, beta)
        noise_sim = NoiseSimulator(self.noise_model_qtensor)
        
        # Simulate
        for num_circs, i in zip(self.num_circs_list, range(len(self.num_circs_list))):
            result = NoiseSimComparisonResult(self.qiskit_circ, self.qtensor_circ, self.noise_model_qiskit, 
                self.noise_model_qtensor, self.n, self.p, self.d)
            if i == 0 or recompute_previous_ensemble == False: 
                self.num_circs_simulated.append(num_circs)
                noise_sim.simulate_batch_ensemble_density(self.qtensor_circ, num_circs, self.n)
                self.qtensor_density_matrix = noise_sim.normalized_ensemble_density_matrix
            else: 
                actual_num_circs = num_circs - self.num_circs_list[i - 1]
                self.num_circs_simulated.append(actual_num_circs)
                noise_sim.simulate_batch_ensemble_density(self.qtensor_circ, actual_num_circs, self.n)
                prev_density_matrix = self.prev_qtensor_density_matrix
                curr_density_matrix = noise_sim.normalized_ensemble_probs
                self.qtensor_density_matrix = (curr_density_matrix + prev_density_matrix) / 2
            qtensor_sim_time = noise_sim.time_taken
            self.simulate_qiskit_density_matrix(self.qiskit_circ, self.noise_model_qiskit, take_trace = False)

            if recompute_previous_ensemble == False: 
                self.prev_qtensor_density_matrix = self.qtensor_density_matrix
            # Save results
            result.save_results_density(self.qiskit_density_matrix, self.qtensor_density_matrix, num_circs, 
                self.num_circs_simulated[i], G, gamma, beta, qtensor_sim_time, self.qiskit_sim_time)
            self.results.append(result.data)


    # Prepare arguments to be sent to each unit of work 
    def _get_args(self, i):
        if i == 0 or self.recompute_previous_ensemble == False:
            num_circs = self.num_circs_list[i]
        else:
            num_circs = self.num_circs_list[i] - self.num_circs_list[i - 1]
        ## We make sure that each job has a minimum number of circuits. We do this because 
        ## if there are too few circuits for each unit of work, the overhead from 
        ## parallelization removes any advantage gained 
        ## TODO: determine a better minimum number of circuits. Currently 10 is chosen arbitrarily
        min_circs_per_job = min(10, num_circs)
        if self.num_nodes * self.num_jobs_per_node > num_circs / min_circs_per_job:
            num_circs_per_job = min_circs_per_job
            total_jobs = int(np.ceil(num_circs / num_circs_per_job))
        else: 
            total_jobs = self.num_nodes * self.num_jobs_per_node
            num_circs_per_job = int(np.ceil(num_circs / total_jobs))

        ## We make sure that regardless of how many nodes and jobs per node we have, we always 
        ## simulate the exact number of circuits in the ensemble specified. 
        if num_circs_per_job * total_jobs == num_circs:
            self._arggen = list(zip(repeat(self.noise_sim, total_jobs), repeat(self.qtensor_circ, total_jobs), 
                repeat(num_circs_per_job, total_jobs), repeat(self.n, total_jobs)))
            self._total_jobs = total_jobs
            self.num_circs_simulated.append(num_circs)
            #print("num_circs: {}, actual num_circs simulated on this iteration: {}".format(self.num_circs_list[i], num_circs))
        else: 
            self._arggen = list(zip(repeat(self.noise_sim, total_jobs - 1), repeat(self.qtensor_circ, total_jobs - 1), 
                repeat(num_circs_per_job, total_jobs - 1), repeat(self.n, total_jobs - 1)))
            num_circs_in_last_job = num_circs % num_circs_per_job
            self._arggen.append((self.noise_sim, self.qtensor_circ, num_circs_in_last_job, self.n))
            self._total_jobs = total_jobs
            actual_num_circs = (total_jobs - 1) * num_circs_per_job + num_circs_in_last_job
            assert num_circs == actual_num_circs
            self.num_circs_simulated.append(actual_num_circs)
            #print("num_circs: {}, actual num_circs simulated on this iteration: {}, total jobs: {}".format(self.num_circs_list[i], actual_num_circs, total_jobs))


    def simulate_qiskit_density_matrix(self, circuit, noise_model_qiskit, take_trace = True):
        start = time.time_ns() / (10 ** 9)
        backend = AerSimulator(method='density_matrix', noise_model=noise_model_qiskit, fusion_enable=False, fusion_verbose=True)
        result = execute(circuit, backend, shots=1).result()
        result = backend.run(circuit).result()
        #result = execute(circuit, Aer.get_backend('aer_simulator_density_matrix'), shots=1, noise_model=noise_model_qiskit).result()
        if take_trace:
            self.qiskit_probs = np.diagonal(result.results[0].data.density_matrix.real)
            end = time.time_ns() / (10 ** 9)
            self.qiskit_sim_time = end - start
        else:
            self.qiskit_density_matrix = result.results[0].data.density_matrix.real
            end = time.time_ns() / (10 ** 9)
            self.qiskit_sim_time = end - start

    def _get_circuits(self, G, gamma, beta):
        # Create Qiskit circuit 
        qiskit_com = QiskitQAOAComposer(graph=G, gamma=gamma, beta=beta)
        qiskit_com.ansatz_state()

        # Convert Qiskit circuit to Qtree circuit
        self.num_qubits, self.qtensor_circ = from_qiskit_circuit(qiskit_com.circuit)
        
        # Finish building remaining portion of Qiskit circuit used only in an Aer simulation 
        qiskit_com.circuit = qiskit_com.circuit.reverse_bits()
        qiskit_com.circuit.save_density_matrix()
        qiskit_com.circuit.measure_all(add_bits = False)
        self.qiskit_circ = qiskit_com.circuit

    def _check_params(self):
        if not isinstance(self.n, int):
            raise Exception("n must an integer.")
        if not isinstance(self.p, int):
            raise Exception("p must an integer.")
        if not isinstance(self.d, int):
            raise Exception("d must an integer.")
        if not isinstance(self.noise_model_qiskit, noise.NoiseModel):
            raise Exception("Qiskit noise model must be of type 'noisel.NoiseModel'")
        if not isinstance(self.noise_model_qtensor, NoiseModel):
            raise Exception("QTensor noise model must be of type NoiseModel.NoiseModel")
        if not isinstance(self.num_circs_list, list):
            raise Exception("The number of circuits must be given as a list. I.e. if num_circs = 10, the argument should be [10].")
        if any(not isinstance(y, int) for y in self.num_circs_list):
            raise Exception("The number of circuits specified must a list of integers.")
        if (self.n * self.d) % 2 != 0:
            raise Exception("n * d must be even. This was not satisfied for the given values d: {}, n: {}".format(self.d, self.n))
        if not 0 <= self.d < self.n:
            raise Exception("The inequality 0 <= d < n was not satisfied for the given values d: {}, n: {}".format(self.d, self.n))

    def _check_correct_num_circs_simulated(self, i):
        if i > 0:
            assert self.num_circs_list[i] == self.num_circs_list[i - 1] + self.num_circs_simulated[i]


