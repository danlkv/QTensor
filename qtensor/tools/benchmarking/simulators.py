import numpy as np
import time
from typing import Union
from qtensor.tools.lazy_import import quimb, acqdp
from qtensor.tools.lazy_import import cotengra as ctg
from qtensor.tools.lightcone_orbits import get_edge_orbits_lightcones
from dataclasses import dataclass
import qtensor.tests.qiskit_qaoa_energy
import qtensor
from qtensor.tests.acqdp_qaoa import qaoa as acqdp_qaoa
from qtensor.tests import qaoa_quimb
import pyrofiler.c as profiles
from tqdm.auto import tqdm


def get_test_gamma_beta(p):
    return [.1]*p, [.6]*p

class BenchCircuit:
    @classmethod
    def from_qsim_stream(cls):
        raise NotImplementedError()

    @classmethod
    def qaoa_ansatz(cls):
        raise NotImplementedError()

    @classmethod
    def qaoa_energy_edge(cls):
        raise NotImplementedError()


class TimeExceeded(RuntimeError):
    """Raised when either optimization time or contraction time exceeded"""
    pass

class MemoryWillExceed(RuntimeError):
    """Raised when width of a contrcaction will be larger than max_width"""
    pass

class BenchSimulator:
    _max_time: float
    _max_width: float
    def __init__(self, max_time=np.inf, max_width=np.inf):
        self._max_time = max_time
        self._max_width = max_width

    def set_max_time(self, max_time: float):
        self._max_time = float(max_time)

    def set_max_width(self, max_width: float):
        self._max_width = float(max_width)

    def _iterate_with_time(self, iterable):
        start = time.time()
        a = 0
        for x in iterable:
            a+= 1
            # don't check time if we have infinite budget
            if self._max_time < np.inf:
                ts = time.time()
                if ts - start < self._max_time:
                    yield x
                else:
                    raise TimeExceeded((
                        f'Iteration time budget was {self._max_time}',
                        f' but already spent {ts-start} seconds'
                    ))
            else:
                yield x

    def optimize_qaoa_energy(self, G, p):
        raise NotImplementedError()

    def simulate_qaoa_energy(self, G, p):
        raise NotImplementedError()

    def optimize(self, circuit):
        raise NotImplementedError()

    def simulate(self, circuit, opt):
        raise NotImplementedError()

    def simulate_prof(self, circuit, opt, *args, **kwargs):
        with profiles.timing() as t:
            with profiles.mem_util() as m:
                res = self.simulate(circuit, opt, *args, **kwargs)
        return res, t.result, m.result

@dataclass
class ContractionEstimation:
    def __init__(self, width, flops, mems):
        # Quimb return a strange Decimal object that doesn't pickle
        self.width = int(width)
        self.flops = int(flops)
        self.mems = int(mems)


class QiskitSimulator(BenchSimulator):
    def optimize_qaoa_energy(self, G, p):
        N = G.number_of_nodes()
        est = ContractionEstimation(
            width=N,
            mems=2**N,
            flops=p*2**N
        )
        return None, [est], 0

    def simulate_qaoa_energy(self, G, p, opt, method='automatic'):
        gamma, beta = [0.1]*p, [.2]*p
        res = 0
        with profiles.timing() as t:
            with profiles.mem_util() as m:
                if G.number_of_nodes()>self._max_width:
                    raise MemoryWillExceed('qiskit: too many qubits')
                res = qtensor.tests.qiskit_qaoa_energy.simulate_qiskit_amps(
                    G, gamma, beta, method=method
                )
        return res, t.result, m.result


class QtensorSimulator(BenchSimulator):
    def __init__(self, backend='einsum', accelerated=False, max_time=None, max_width=None):
        self.backend = backend
        self.accelerated = accelerated
        self.max_time = max_time
        self.max_width = max_width

    def _iterate_edges_accelerated(self, G, p):
        with profiles.timing() as t:
            eorbits, _ = get_edge_orbits_lightcones(G,p)
        print('eorbits time', t.result)

        iterator = [x[0] for x in eorbits.values()]
        return self._iterate_with_time(iterator)

    def _iterate_edges(self, G, p):
        return self._iterate_with_time(G.edges)

    def iterate_edges(self, G, p):
        if self.accelerated:
            gen = self._iterate_edges_accelerated(G, p)
        else:
            gen = self._iterate_edges(G, p)
        return tqdm(gen, total=G.number_of_edges())

    def _get_simulator(self):
        backend = self.backend
        return qtensor.QAOAQtreeSimulator(
            qtensor.DefaultQAOAComposer,
            backend=qtensor.contraction_backends.get_backend(backend)
        )

    def optimize_qaoa_energy(self, G, p,
                             ordering_algo='greedy',
                             opt_kwargs={}
                            ):
        opt = qtensor.toolbox.get_ordering_algo(ordering_algo, **opt_kwargs)
        gamma, beta = get_test_gamma_beta(p)
        # convert gamma beta to qtensor format
        gamma = -np.array(gamma)/np.pi
        beta = np.array(beta)/np.pi
        sim = self._get_simulator()
        opt_time = 0
        ests = []
        opts = []
        for edge in self.iterate_edges(G, p):
            with profiles.timing() as t:
                circuit = sim._edge_energy_circuit(G, gamma, beta, edge)
                tn = qtensor.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(circuit)
                peo, _ = opt.optimize(tn)
            opt_time += t.result
            mems, flops = tn.simulation_cost(peo)
            ests.append(ContractionEstimation(
                width=opt.treewidth,
                mems=16*max(mems),
                flops=sum(flops)
            ))
            opts.append(peo)
        return opts, ests, opt_time

    def simulate_qaoa_energy(self, G, p, opt):
        gamma, beta = get_test_gamma_beta(p)
        # convert gamma beta to qtensor format
        gamma = np.array(gamma)/np.pi
        beta = np.array(beta)/np.pi

        sim = self._get_simulator()
        res = 0
        with profiles.timing() as t:
            with profiles.mem_util() as m:
                for edge, peo in zip(self.iterate_edges(G, p), opt):
                    circuit = sim._edge_energy_circuit(G, gamma, beta, edge)
                    res += sim.simulate_batch(circuit, peo=peo)
        return res, t.result, m.result

    def optimize(self, circuit, ordering_algo='greedy'):
        opt = qtensor.toolbox.get_ordering_algo(ordering_algo)
        with profiles.timing() as t:
            tn = qtensor.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(circuit)
            peo, _ = opt.optimize(tn)
        mems, flops = tn.simulation_cost(peo)
        est = ContractionEstimation(
            width=opt.treewidth,
            flops=sum(flops),
            mems=16*max(mems)
        )
        return peo, est, t.result

    def simulate(self, circuit, opt, backend='einsum'):
        sim = qtensor.QtreeSimulator(
            backend=qtensor.contraction_backends.get_backend(backend)
        )
        res = sim.simulate_batch(circuit, peo=opt)
        return res

class MergedQtensorSimulator(QtensorSimulator):

    def _get_simulator(self):
        backend = self.backend
        return qtensor.MergedSimulator.MergedQAOASimulator(
            qtensor.DefaultQAOAComposer,
            backend=qtensor.contraction_backends.get_backend(backend)
        )

    def optimize_qaoa_energy(self, G, p,
                             ordering_algo='greedy',
                             opt_kwargs={}
                            ):
        opt = qtensor.toolbox.get_ordering_algo(ordering_algo, **opt_kwargs)
        gamma, beta = get_test_gamma_beta(p)
        # convert gamma beta to qtensor format
        gamma = np.array(gamma)/np.pi
        beta = np.array(beta)/np.pi
        sim = self._get_simulator()
        sim.optimizer = opt
        opt_time = 0
        ests = []
        peos = []
        for edge in self.iterate_edges(G, p):
            with profiles.timing() as t:
                circuit = sim._edge_energy_circuit(G, gamma, beta, edge)
                peo, width = sim.simulate_batch(circuit, dry_run=True)
            opt_time += t.result
            peos.append((sim.ibunch, sim.merged_buckets))

            #tn = qtensor.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(circuit)
            #mems, flops = tn.simulation_cost(peo)
            ests.append(ContractionEstimation(
                width=width,
                mems=16*2**width,
                flops=2**(width +1)
            ))
        return peos, ests, opt_time

    def simulate_qaoa_energy(self, G, p, opt):
        gamma, beta = get_test_gamma_beta(p)
        # convert gamma beta to qtensor format
        gamma = np.array(gamma)/np.pi
        beta = np.array(beta)/np.pi
        sim = self._get_simulator()
        result = 0
        with profiles.timing() as t:
            with profiles.mem_util() as m:
                # should not consume all iterator at once
                for edge, (ibunch, merged_buckets) in zip(self.iterate_edges(G, p), opt):
                    edge_contribution = qtensor.merged_indices.bucket_elimination(
                        [x.copy() for x in merged_buckets],
                        ibunch,
                        sim.backend.process_bucket_merged,
                        n_var_nosum=0
                    )
                    result += edge_contribution.data.flatten()

        return result, t.result, m.result


class AcqdpSimulator(BenchSimulator):
    def __init__(self, ordering_algo='oe', order_finder_params={}):
        self.ordering_algo = ordering_algo
        self.order_finder_params = order_finder_params

    def optimize_qaoa_energy(self, G, p,
                             simp_kwargs={}, **kwargs):
        a = {e: [[1,-1],[-1,1]] for e in G.edges}
        gamma, beta = get_test_gamma_beta(p)
        beta_gamma = np.array(beta + gamma)
        # make sure to not throw an error at optimization
        # by artificially setting a larger memory (default = 16)
        compiler_params = dict( memory=16 * 2**20 )
        with profiles.timing() as t:
            self.q = acqdp_qaoa.QAOAOptimizer(a, num_layers=p)
            # Wrap clauses in decorator that will raise on time budget exceeded
            # This requires understanding qtensor/tests/acqdp_qaoa/qaoa.py
            # store the clauses object for transparent patching
            _clauses = self.q.clauses
            self.q.clauses = self._iterate_with_time(self.q.clauses)
            self.q.preprocess(
                order_finder_name=self.ordering_algo,
                order_finder_params=self.order_finder_params,
                compiler_params=compiler_params
            )
            # restore tha clauses object. The self.q might be serialized
            self.q.clauses = _clauses

        ests = []
        for flops, mems in self.q.lightcone_flops_mems:
            ests.append(ContractionEstimation(
                width=np.log2(float(mems)),
                flops=flops,
                mems=16*mems
            ))
        return self.q, ests, t.result

    def simulate_qaoa_energy(self, G, p, opts,
                             simp_kwargs={}, **kwargs):
        gamma, beta = get_test_gamma_beta(p)
        beta_gamma = np.array(beta + gamma)
        res = 0
        with profiles.timing() as t:
            with profiles.mem_util() as m:
                # Wrap clauses in decorator that will raise on time budget exceeded
                # This requires understanding qtensor/tests/acqdp_qaoa/qaoa.py
                class FancyTimeDict(dict):
                    def __iter__(self2):
                        iter_orig = super().__iter__()
                        return self._iterate_with_time(iter_orig)

                opts.query_dict = FancyTimeDict(opts.query_dict.items())
                res = opts.query(params=beta_gamma)
        return res, t.result, m.result


class QuimbSimulator(BenchSimulator):
    def __init__(self, simplify_sequence='ADCRS',
                 opt_kwargs=dict(max_repeats=3),
                 opt_type='uniform'):
        self.simplify_sequence = simplify_sequence
        self.opt_kwargs = opt_kwargs
        self.opt_type = opt_type

    def _get_optimizer(self, opt_type=None, **kwargs):
        kwargs = {**self.opt_kwargs, **kwargs}
        if opt_type is None:
            opt_type = self.opt_type

        if opt_type == 'hyper':
            optimizer = ctg.HyperOptimizer(
                parallel=False,
                **kwargs
            )
        elif opt_type == 'uniform':
            optimizer = ctg.UniformOptimizer(
                parallel=False,
                **kwargs
            )
        else:
            raise ValueError('Opt type not supported! Received {} should be one of `hyper` or `uniform`'.format(opt_type))
        return optimizer

    def _iterate_edges(self, G):
        return self._iterate_with_time(G.edges)

    def optimize(self, circuit, opt_type=None, simp_kwargs={}, **kwargs):
        optimizer = self._get_optimizer(opt_type=opt_type, **kwargs)
        simp_kwargs['simplify_sequence'] = simp_kwargs.pop(
            'simplify_sequence', self.simplify_sequence)

        with profiles.timing() as t:
            rehs = circuit.amplitude_rehearse(optimize=optimizer,
                                              **simp_kwargs)

        est = self._rehs2est(rehs)
        return (rehs['info'], rehs['tn']), est, t.result


    def optimize_qaoa_energy(self, G, p, opt_type=None,
                             simp_kwargs=None, **kwargs):
        circuit = self._qaoa_circ(G, p)
        infos = []
        ests = []
        times = []
        if simp_kwargs is None:
            simp_kwargs = {}
        simp_kwargs['simplify_sequence'] = simp_kwargs.pop(
            'simplify_sequence', self.simplify_sequence)
        print('simp kw', simp_kwargs, self.simplify_sequence)

        for edge in tqdm(self._iterate_edges(G), total=G.number_of_edges()):
            with profiles.timing() as t:
                #print('qmb kwargs', kwargs)
                optimizer = self._get_optimizer(opt_type=opt_type, **kwargs)
                ZZ = quimb.pauli('Z') & quimb.pauli('Z')
                rehs = circuit.local_expectation_rehearse(
                    ZZ, edge, optimize=optimizer, **simp_kwargs)

            times.append(t.result)
            infos.append((rehs['info'], rehs['tn']))
            ests.append(self._rehs2est(rehs))

        return infos, ests, sum(times)


    def simulate_qaoa_energy(self, G, p, opts,
                             simp_kwargs={}, **kwargs):
        res = 0
        circuit = self._qaoa_circ(G, p)
        print('simulating energy')
        ZZ = quimb.pauli('Z') & quimb.pauli('Z')
        with profiles.timing() as t:
            with profiles.mem_util() as m:
                for edge, opt in tqdm(
                    zip(self._iterate_edges(G), opts),
                    total=G.number_of_edges()
                ):
                 #  ## --
                 #  # The process of generating TN is for some reason
                 #  # non-deterministic, i.e. the simplification result can be 
                 #  # different for same arguments. Because of that caching should be
                 #  # done on TN level, not on the circuit level.
                 #  #
                 #  # Hovewer, in real-world simulations, when \gamma, \beta changes
                 #  # one will have to re-generate the TN (the tensor data is generated
                 #  # in Tensor.gate() method, on-the-fly). Hence, this dummy line.
                 #  _ = qaoa_quimb.get_lightcone_tn(circuit, ZZ,
                 #                              where=edge,
                 #                              simplify_sequence=self.simplify_sequence
                 #                             )
                    ## --
                    info, tn = opt
                    path = info.path
                    res += tn.contract(output_inds=(), optimize=path, **kwargs)

        return res, t.result, m.result


    def simulate(self, circuit, opt,
                 **kwargs)  :
        info = opt['info']
        tn = opt['tn']
        return tn.contract(output_inds=(), optimize=info.path, **kwargs)

    def _rehs2est(self, rehs):
        tree = ctg.ContractionTree.from_info(rehs['info'])
        return ContractionEstimation(
            width = rehs['W'],
            flops = tree.total_flops(),
            mems = 16*rehs['info'].largest_intermediate
        )

    def _qaoa_circ(self, G, p):
        terms = {(i, j):1 for i, j in G.edges}
        gammas, betas = get_test_gamma_beta(p)
        return quimb.tensor.circ_qaoa(terms, p, gammas, betas)


SIMULATORS = {
    'quimb': QuimbSimulator,
    'qtensor': QtensorSimulator,
    'qtensor_merged': MergedQtensorSimulator,
    'acqdp': AcqdpSimulator,
}
