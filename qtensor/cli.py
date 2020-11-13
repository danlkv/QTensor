import sys
import time
import click
import qtree
import networkx as nx
import numpy as np
import random
from tqdm import tqdm

import qtree.operators as ops
import qtensor.optimisation as qop
from qtensor.FeynmanSimulator import FeynmanSimulator

from qtensor.ProcessingFrameworks import PerfNumpyBackend
from qtensor.toolbox import qaoa_energy_tw_from_graph, get_ordering_algo
from qtensor.optimisation.TensorNet import QtreeTensorNet
from qtensor.optimisation.Optimizer import OrderingOptimizer, TamakiOptimizer, WithoutOptimizer
from qtensor.ProcessingFrameworks import PerfBackend, CMKLExtendedBackend
from qtensor.optimisation.Optimizer import TamakiTrimSlicing, SlicesOptimizer
from qtensor import DefaultQAOAComposer, QAOAQtreeSimulator
import qtensor.ProcessingFrameworks as backends
import qtensor.optimisation.Optimizer as optimizers
from qtensor.optimisation import RGreedyOptimizer

@click.group()
def cli():
    pass

def choose_backend(backend_str):
    if backend_str=='numpy':
        return backends.NumpyBackend
    elif backend_str=='mkl':
        return backends.CMKLExtendedBackend
    elif backend_str=='exatn':
        return backends.ExaTnBackend

@cli.command()
@click.argument('filename', nargs=-1)
@click.option('-p','--num-processes', default=1)
@click.option('-P','--profile', default=False, is_flag=True)
@click.option('-t','--max-tw', default=25)
@click.option('-B','--backend', default='numpy')
@click.option('-O','--optimizer', default='greedy')
def sim_file(filename, profile=False, num_processes=1, max_tw=25, backend='numpy', optimizer='greedy'):
    if not filename:
        stream = sys.stdin
    else:
        stream = open(filename[0],'r')

    n_qubits, circuit = ops.read_circuit_stream(stream)
    kwargs = dict(
        n_processes=num_processes
        ,max_tw=max_tw
        , pool_type='thread'
    )
    Backend = choose_backend(backend)
    if profile:
        class DynamicallyGeneratedBackend(PerfBackend):
            Backend = Backend
        backend_obj = DynamicallyGeneratedBackend(print=False)
        kwargs['bucket_backend'] = backend_obj
    else:
        kwargs['bucket_backend'] = Backend()

    if optimizer=='tamaki':
        kwargs['optimizer'] = TamakiTrimSlicing(max_tw=max_tw, wait_time=23)
    else:
        kwargs['optimizer'] = SlicesOptimizer(max_tw=max_tw, tw_bias=0)
    kwargs['optimizer'].max_tw = max_tw


    sim = FeynmanSimulator(**kwargs)
    circuit = sum(circuit, [])
    result = sim.simulate(circuit, batch_vars=0, tw_bias=0)
    print(result)

    if profile:
        print('Profiling results')
        backend_obj.gen_report()

@cli.command()
@click.argument('filename', nargs=-1)
@click.option('-t', '--tamaki-time', default=15)
@click.option('-T', '--max-tw', default=32)
@click.option('-s', '--slice-step', default=None, type=int)
@click.option('-C', '--cost-type', default='length')
def opt_file(filename, tamaki_time, max_tw, slice_step, cost_type):
    if not filename:
        stream = sys.stdin
    else:
        stream = open(filename[0],'r')

    n_qubits, circuit = ops.read_circuit_stream(stream)
    gates = sum(circuit, [])
    tn = qop.TensorNet.QtreeTensorNet.from_qtree_gates(gates)
    fopt = qop.Optimizer.TamakiTrimSlicing(wait_time=tamaki_time)
    fopt.max_tw = max_tw
    fopt.par_var_step = slice_step
    fopt.cost_type = cost_type
    fopt.tw_bias = 0
    try:
        peo, par_vars, tn = fopt.optimize(tn)
        #print('peo', peo)
    except Exception as e:
        print(e)

    hist = fopt._slice_hist
    sep = '\t'

    print(sep.join(['p_vars','tw']))
    for x in hist:
        print(sep.join(str(n) for n in x))


@cli.command()
@click.argument('filename')
def tw_exact(filename):
    tn = qop.TensorNet.QtreeTensorNet.from_qsim_file(filename)
    graph = tn.get_line_graph()
    peo, tw = qtree.graph_model.peo_calculation.get_peo(graph)
    print(peo)
    print(tw)


@cli.command()
@click.argument('filename')
@click.option('-t','--tamaki_time', default=15)
def tw_heuristic(filename, tamaki_time):
    tn = qop.TensorNet.QtreeTensorNet.from_qsim_file(filename)
    fopt = qop.Optimizer.TamakiOptimizer()
    fopt.wait_time = tamaki_time
    try:
        peo, tn = fopt.optimize(tn)
        data = {'treewidth': fopt.treewidth}
        print(data)
        #print('peo', peo)
    except Exception as e:
        print(e)

@cli.command()
@click.option('-s','--seed', default=42)
@click.option('-d','--degree', default=3)
@click.option('-n','--nodes', default=10)
@click.option('-p','--p', default=1)
@click.option('-O','--ordering-algo', default='greedy')
@click.option('-G','--graph-type', default='random_regular')
def optimize_qaoa_ansatz_circuit(seed, degree, nodes, p, graph_type, ordering_algo):
    np.random.seed(seed)
    random.seed(seed)
    if graph_type=='random_regular':
        G = nx.random_regular_graph(degree, nodes)
    elif graph_type=='erdos_renyi':
        G = nx.erdos_renyi_graph(nodes, degree/(nodes-1))
    elif graph_type=='erdos_renyi_core':
        G = nx.erdos_renyi_graph(nodes, degree/(nodes-1))
        G = nx.algorithms.core.k_core(G, k=degree)
    else:
        raise Exception('Unsupported graph type')

    if ordering_algo=='tamaki_slice':
        optimizer = TamakiTrimSlicing(max_tw=max_tw, wait_time=tamaki_time)
    elif ordering_algo=='tamaki':
        tamaki_time = 130
        optimizer = optimizers.TamakiOptimizer(wait_time=tamaki_time)
    elif 'rgreedy' in ordering_algo:
        if '_' in ordering_algo:
            params = ordering_algo.split('_')
            if len(params) == 2:
                _, temp = ordering_algo.split('_')
                repeats = 10
            else:
                _, temp, repeats = ordering_algo.split('_')
            repeats = int(repeats)
            temp = float(temp)
        else:
            temp = 2
            repeats = 10
        optimizer = RGreedyOptimizer(temp=temp, repeats=repeats)
    elif ordering_algo == 'greedy':
        optimizer = optimizers.DefaultOptimizer()
    else:
        raise ValueError('Ordering algorithm not supported')
    gamma, beta = [0.1]*p, [0.2]*p
    composer = DefaultQAOAComposer(G, beta=beta, gamma=gamma)
    composer.ansatz_state()
    tn = qop.TensorNet.QtreeTensorNet.from_qtree_gates(composer.circuit)
    print(optimizer)
    optimizer.optimize(tn)
    print('tw=', optimizer.treewidth)

@cli.command()
@click.option('-s','--seed', default=42)
@click.option('-d','--degree', default=3)
@click.option('-n','--nodes', default=10)
@click.option('-p','--p', default=1)
@click.option('-G','--graph-type', default='random_regular')
def generate_qaoa_ansatz_circuit(seed, degree, nodes, p, graph_type):
    np.random.seed(seed)
    random.seed(seed)
    if graph_type=='random_regular':
        G = nx.random_regular_graph(degree, nodes)
    elif graph_type=='erdos_renyi':
        G = nx.erdos_renyi_graph(nodes, degree/(nodes-1))
    elif graph_type=='erdos_renyi_core':
        G = nx.erdos_renyi_graph(nodes, degree/(nodes-1))
        G = nx.algorithms.core.k_core(G, k=degree)
    else:
        raise Exception('Unsupported graph type')
    gamma, beta = [0.1]*p, [0.2]*p
    composer = DefaultQAOAComposer(G, beta=beta, gamma=gamma)
    composer.ansatz_state()
    txt = qtree.operators.circuit_to_text([composer.circuit], nodes)
    print(txt)

@cli.command()
@click.option('-s','--seed', default=42)
@click.option('-d','--degree', default=3)
@click.option('-n','--nodes', default=10)
@click.option('-p','--p', default=1)
@click.option('-G','--graph-type', default='random_regular')
@click.option('-E','--edge-index', default=0)
def generate_qaoa_energy_circuit(seed, degree, nodes, p, graph_type, edge_index):
    np.random.seed(seed)
    random.seed(seed)
    if graph_type=='random_regular':
        G = nx.random_regular_graph(degree, nodes)
    elif graph_type=='erdos_renyi':
        G = nx.erdos_renyi_graph(nodes, degree/(nodes-1))
    elif graph_type=='erdos_renyi_core':
        G = nx.erdos_renyi_graph(nodes, degree/(nodes-1))
        G = nx.algorithms.core.k_core(G, k=degree)
    else:
        raise Exception('Unsupported graph type')
    gamma, beta = [0.1]*p, [0.2]*p
    edge = list(G.edges())[edge_index]
    composer = DefaultQAOAComposer(G, beta=beta, gamma=gamma)
    composer.energy_expectation_lightcone(edge)
    txt = qtree.operators.circuit_to_text([composer.circuit], composer.n_qubits)
    print(txt)

@cli.command()
@click.option('-s','--seed', default=42)
@click.option('-d','--degree', default=3)
@click.option('-n','--nodes', default=10)
@click.option('-p','--p', default=1)
@click.option('-G','--graph-type', default='random_regular')
@click.option('-T','--max-time', default=0, help='Max time for every evaluation')
@click.option('--max-tw', default=0, help='Max tw after wich no point to calculate')
@click.option('-O','--ordering-algo', default='greedy', help='Algorithm for elimination order')
@click.option('--tamaki_time', default=20, help='Algorithm for elimination order')
@click.option('--n_processes', default=1, help='Number of processes.')
def qaoa_energy_tw(nodes, seed, degree, p, graph_type, max_time, max_tw, ordering_algo, tamaki_time, n_processes):
    np.random.seed(seed)
    random.seed(seed)
    if graph_type=='random_regular':
        G = nx.random_regular_graph(degree, nodes)
    elif graph_type=='erdos_renyi':
        G = nx.erdos_renyi_graph(nodes, degree/(nodes-1))
    else:
        raise Exception('Unsupported graph type')

    qaoa_energy_tw_from_graph(G, p, max_time, max_tw, ordering_algo, print_stats=True, tamaki_time=tamaki_time, n_processes=n_processes)


@cli.command()
@click.option('-s','--seed', default=42)
@click.option('-d','--degree', default=3)
@click.option('-n','--nodes', default=10)
@click.option('-p','--p', default=1)
@click.option('-G','--graph-type', default='random_regular')
@click.option('-T','--max-time', default=0, help='Max time for every evaluation')
@click.option('--max-tw', default=0, help='Max tw after wich no point to calculate')
@click.option('-O','--ordering-algo', default='greedy', help='Algorithm for elimination order')
@click.option('-B','--backend', default='numpy')
@click.option('--n_processes', default=1)
@click.option('-P','--profile', default=False, is_flag=True)
@click.option('-C','--composer-type', default='default')
def qaoa_energy_sim(nodes, seed,
                    degree, p, graph_type,
                    max_time, max_tw, ordering_algo,
                    backend, n_processes, profile,
                    composer_type='default'):
    np.random.seed(seed)
    random.seed(seed)
    if graph_type=='random_regular':
        G = nx.random_regular_graph(degree, nodes, seed=seed)
    elif graph_type=='erdos_renyi':
        G = nx.erdos_renyi_graph(nodes, degree/(nodes-1), seed=seed)
    else:
        raise Exception('Unsupported graph type')
    gamma, beta = [np.pi/3]*p, [np.pi/2]*p

    optimizer = get_ordering_algo(ordering_algo)

    Backend = choose_backend(backend)
    backend_obj = Backend()
    if profile:
        backend_obj = PerfBackend(print=False)
        backend_obj.backend = Backend()

    sim = QAOAQtreeSimulator(DefaultQAOAComposer, bucket_backend=backend_obj, optimizer=optimizer)
    start = time.time()
    if n_processes==1:
        result = sim.energy_expectation(G, gamma, beta)
        if profile:
            print('Profiling results')
            backend_obj.gen_report()
    else:
        result = sim.energy_expectation_parallel(G, gamma, beta, n_processes=n_processes)
    end = time.time()
    print(f"Simutation time: {end - start}")
    print(result)


cli()
