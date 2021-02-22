import cotengra as ctg
import numpy as np
import quimb
import acqdp
import qtensor
from qtensor.tests.acqdp_qaoa import qaoa as acqdp_qaoa
import pyrofiler.c as profiles
from tqdm.auto import tqdm


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

class BenchSimulator:
    def optimize_qaoa_expectation(self, G, p):
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

class ContractionEstimation:
    def __init__(self, width, flops, mems):
        self.width = width
        self.flops=flops
        self.mems = mems


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
                if G.number_of_nodes()>26:
                    raise Exception('qiskit: too many qubits')
                res = qtensor.tests.qiskit_qaoa_energy.simulate_qiskit_amps(
                    G, gamma, beta, method=method
                )
        return res, t.result, m.result


class QtensorSimulator(BenchSimulator):
    def optimize_qaoa_energy(self, G, p,
                                  merged=False,
                                  ordering_algo='greedy',

                                 ):
        opt = qtensor.toolbox.get_ordering_algo(ordering_algo)
        gamma, beta = [0.1]*p, [.2]*p
        sim = qtensor.QAOAQtreeSimulator(qtensor.DefaultQAOAComposer)
        opt_time = 0
        ests = []
        peos = []
        for edge in tqdm(G.edges):
            with profiles.timing() as t:
                circuit = sim._edge_energy_circuit(G, gamma, beta, edge)
                tn = qtensor.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(circuit)
                peo, _ = opt.optimize(tn)
            opt_time += t.result
            mems, flops = tn.simulation_cost(peo)
            ests.append(ContractionEstimation(
                width=opt.treewidth,
                mems=max(mems),
                flops=sum(flops)
            ))
            peos.append(peo)
        return peos, ests, opt_time

    def simulate_qaoa_energy(self, G, p, opt, backend='greedy'):
        gamma, beta = [0.1]*p, [.2]*p
        sim = qtensor.QAOAQtreeSimulator(
            qtensor.DefaultQAOAComposer,
            backend=qtensor.contraction_backends.get_backend(backend)
        )
        res = 0
        with profiles.timing() as t:
            with profiles.mem_util() as m:
                for edge, peo in tqdm(zip(G.edges, opt)):
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
            mems=max(mems)
        )
        return peo, est, t.result

    def simulate(self, circuit, opt, backend='einsum'):
        sim = qtensor.QtreeSimulator(
            backend=qtensor.contraction_backends.get_backend(backend)
        )
        res = sim.simulate_batch(circuit, peo=opt)
        return res


class AcqdpSimulator(BenchSimulator):
    def optimize_qaoa_energy(self, G, p, ordering_algo='oe',
                             simp_kwargs={}, **kwargs):
        a = {e: [[1,-1],[-1,1]] for e in G.edges}
        beta_gamma = np.random.randn(p*2)
        with profiles.timing() as t:
            self.q = acqdp_qaoa.QAOAOptimizer(a, num_layers=p)
            self.q.preprocess(order_finder_name=ordering_algo)

        ests = []
        for flops, mems in self.q.lightcone_flops_mems:
            ests.append(ContractionEstimation(
                width=np.log2(float(mems)),
                flops=flops,
                mems=mems
            ))
        return self.q, ests, t.result

    def simulate_qaoa_energy(self, G, p, opts,
                             simp_kwargs={}, **kwargs):
        beta_gamma = np.random.randn(p*2)
        res = 0
        with profiles.timing() as t:
            with profiles.mem_util() as m:
                res = opts.query(params=beta_gamma)
        return res, t.result, m.result


class QuimbSimulator(BenchSimulator):
    def optimize(self, circuit, opt_type='hyper', simp_kwargs={}, **kwargs):
        optimizer = ctg.HyperOptimizer(
            parallel=False,
            **kwargs
        )

        with profiles.timing() as t:
            rehs = circuit.amplitude_rehearse(optimize=optimizer,
                                              **simp_kwargs)

        est = self._rehs2est(rehs)
        return (rehs['info'], rehs['tn']), est, t.result


    def optimize_qaoa_energy(self, G, p, opt_type='hyper',
                             simp_kwargs={}, **kwargs):
        circuit = self._qaoa_circ(G, p)
        infos = []
        ests = []
        times = []
        for edge in tqdm(G.edges):
            with profiles.timing() as t:
                #print('qmb kwargs', kwargs)
                optimizer = ctg.HyperOptimizer(
                    parallel=False,
                    **kwargs
                )
                ZZ = quimb.pauli('Z') & quimb.pauli('Z')
                rehs = circuit.local_expectation_rehearse(
                    ZZ, edge, optimize=optimizer, **simp_kwargs)

            times.append(t.result)
            infos.append((rehs['info'], circuit))
            ests.append(self._rehs2est(rehs))

        return infos, ests, sum(times)


    def simulate_qaoa_energy(self, G, p, opts,
                             simp_kwargs={}, **kwargs):
        circuit = self._qaoa_circ(G, p)
        res = 0
        print('simulating energy')
        with profiles.timing() as t:
            with profiles.mem_util() as m:
                for edge, opt in tqdm(zip(G.edges, opts)):
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
            mems = rehs['info'].largest_intermediate
        )

    def _qaoa_circ(self, G, p):
        terms = {(i, j):1 for i, j in G.edges}
        gammas, betas = [0.1]*p, [.2]*p
        return quimb.tensor.circ_qaoa(terms, p, gammas, betas)
