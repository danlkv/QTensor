from xml.dom.minicompat import NodeList
from NoiseChannels import *
from NoiseSimulator import * 
from NoiseModel import *
from helper_functions import *
from qtensor.tests.test_composers import *
from qtensor.tools import mpi
from itertools import repeat

## TODO
# Save data somewhere (and figure out what exactly to save)
# potentially refactor comparison simulator to inherit this and/or call this. remove duplicate code

class QAOANoiseSimulator:
    def __init__(self):
        pass

    def noisy_qaoa_sim(self, n: int, p: int, d: int, num_circs: int, noise_model: NoiseModel):
        self._set_params(n=n, p=p, d=d, num_circs=num_circs, noise_model=noise_model)
        G, gamma, beta = get_qaoa_params(n=self.n, p=self.p, d=self.d)
        composer = QtreeQAOAComposer(graph=G, gamma=gamma, beta=beta)
        composer.ansatz_state()
        self.circuit = composer.circuit
        noise_sim = NoiseSimulator(self.noise_model)
        noise_sim.simulate_batch_ensemble(self.circuit, self.num_circs, self.num_qubits)
        self.probs = noise_sim.normalized_ensemble_probs
        self.sim_time = noise_sim.time_taken

    def _mpi_parallel_unit(self, args):
        noise_sim, qtensor_circ, num_circs, num_qubits = args
        noise_sim.simulate_batch_ensemble(qtensor_circ, num_circs, num_qubits)
        fraction_of_qtensor_probs = noise_sim.normalized_ensemble_probs
        return fraction_of_qtensor_probs

    def noisy_qaoa_sim_mpi(self, n: int, p: int, d: int, num_circs: int, noise_model: NoiseModel, 
            num_nodes: int, num_jobs_per_node: int, print_stats: bool=False, pbar: bool=True):
            
        self._set_params(mpi_sim=True, n=n, p=p, d=d, num_circs=num_circs, noise_model=noise_model, 
            num_nodes=num_nodes, num_jobs_per_node=num_jobs_per_node, print_stats=print_stats, pbar=pbar)
        G, gamma, beta = get_qaoa_params(n=self.n, p=self.p, d=self.d)
        composer = QtreeQAOAComposer(graph=G, gamma=gamma, beta=beta)
        composer.ansatz_state()
        self.circuit = composer.circuit
        self.noise_sim = NoiseSimulator(self.noise_model)
        arggen = self._get_args(self.num_circs, self.num_nodes, self.num_jobs_per_node)
        qaoa_probs_list = mpi.mpi_map(self._mpi_parallel_unit, arggen, pbar=pbar, total=self.num_nodes)

        if qaoa_probs_list:
            self.probs = sum(qaoa_probs_list) / self.total_jobs
            print(self.probs)
            #self.mpi_sim_time = mpi.get_stats()
            if print_stats:
                mpi.print_stats()

    def _get_args(self, num_circs, num_nodes, num_jobs_per_node):
        total_jobs, num_circs_per_job = self._get_total_jobs(num_circs, num_nodes, num_jobs_per_node)
        """
        We make sure that regardless of how many nodes and jobs per node we have, 
        we always simulate the exact number of circuits in the ensemble specified. 
        """
        if num_circs_per_job * total_jobs == num_circs:
            arggen = list(zip(repeat(self.noise_sim, total_jobs), repeat(self.circuit, total_jobs), 
                repeat(num_circs_per_job, total_jobs), repeat(self.num_qubits, total_jobs)))
        else: 
            arggen = list(zip(repeat(self.noise_sim, total_jobs - 1), repeat(self.circuit, total_jobs - 1), 
                repeat(num_circs_per_job, total_jobs - 1), repeat(self.n, total_jobs - 1)))
            num_circs_in_last_job = num_circs % num_circs_per_job
            arggen.append((self.noise_sim, self.circuit, num_circs_in_last_job, self.n))
            actual_num_circs = (total_jobs - 1) * num_circs_per_job + num_circs_in_last_job
            assert num_circs == actual_num_circs
        self.total_jobs = total_jobs
        return arggen

    def _get_total_jobs(self, num_circs, num_nodes, num_jobs_per_node):
        """
        We make sure that each job has a minimum number of circuits. 
        We do this because if there are too few circuits for each unit of work, 
        the overhead from parallelization removes any advantage gained

        TODO: determine a better minimum number of circuits. Currently 10 is chosen arbitrarily
        """
        min_circs_per_job = min(10, num_circs)
        if num_nodes * num_jobs_per_node > num_circs / min_circs_per_job:
            num_circs_per_job = min_circs_per_job
            total_jobs = int(np.ceil(num_circs / num_circs_per_job))
        else: 
            total_jobs = num_nodes * num_jobs_per_node
            num_circs_per_job = int(np.ceil(num_circs / total_jobs))
        return total_jobs, num_circs_per_job

    def _set_params(self, mpi_sim = False, **params):
        self.n = params['n']
        self.p = params['p']
        self.d = params['d']
        self.num_circs = params['num_circs']
        self.noise_model = params['noise_model']
        self.num_qubits = params['n']
        if mpi_sim:
            self.num_nodes = params['num_nodes']
            self.num_jobs_per_node = params['num_jobs_per_node']
        self._check_params(mpi_sim, params)

    def _check_params(self, mpi_sim, params):
        if not isinstance(self.n, int):
            raise Exception("n must be an integer.")
        if not isinstance(self.p, int):
            raise Exception("p must be an integer.")
        if not isinstance(self.d, int):
            raise Exception("d must be an integer.")
        if not isinstance(self.num_circs, int):
            raise Exception("num_circs must be an integer")
        if not isinstance(self.noise_model, NoiseModel):
            raise Exception("noise_model must be of type NoiseModel.NoiseModel")
        if (self.n * self.d) % 2 != 0:
            raise Exception("n * d must be even. This was not satisfied for the given values d: {}, n: {}".format(self.d, self.n))
        if not 0 <= self.d < self.n:
            raise Exception("The inequality 0 <= d < n was not satisfied for the given values d: {}, n: {}".format(self.d, self.n))
        if mpi_sim:
            if not isinstance(self.num_nodes, int):
                raise Exception("num_nodes must be an integer")
            if not isinstance(self.num_jobs_per_node, int):
                raise Exception("num_jobs_per_node must be an integer")
            if not isinstance(params['print_stats'], bool):
                raise Exception("print_stats must be a bool")
            if not isinstance(params['pbar'], bool):
                raise Exception("pbar must be a bool")
        
