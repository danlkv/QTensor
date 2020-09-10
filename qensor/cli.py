import sys
import time
import click
import qtree
import networkx as nx
import numpy as np
from tqdm import tqdm

import qtree.operators as ops
import qensor.optimisation as qop
from qensor.FeynmanSimulator import FeynmanSimulator

from qensor.optimisation.TensorNet import QtreeTensorNet
from qensor.optimisation.Optimizer import OrderingOptimizer, TamakiOptimizer, WithoutOptimizer
from qensor import QtreeQAOAComposer
from qensor import PerfNumpyBackend
from qensor.ProcessingFrameworks import CMKLExtendedBackend, PerfBackend

@click.group()
def cli():
    pass

@cli.command()
@click.argument('filename', nargs=-1)
@click.option('-p','--num-processes', default=1)
@click.option('-P','--profile', default=False, is_flag=True)
@click.option('-t','--target-tw', default=25)
def sim_file(filename, profile=False, num_processes=1, target_tw=25):
    if not filename:
        stream = sys.stdin
    else:
        stream = open(filename[0],'r')

    n_qubits, circuit = ops.read_circuit_stream(stream)
    kwargs = dict(
        n_processes=num_processes
        ,target_tw=target_tw
        , pool_type='thread'
    )
    if profile:
        class PerfMKLBackend(PerfBackend):
            Backend = CMKLExtendedBackend
        backend = PerfMKLBackend(print=False)
        # backend = PerfNumpyBackend(print=False)
        kwargs['bucket_backend'] = backend
    sim = FeynmanSimulator(**kwargs)
    circuit = sum(circuit, [])
    result = sim.simulate(circuit, batch_vars=4, tw_bias=0)
    print(result)

    if profile:
        print('Profiling results')
        backend.gen_report()

@cli.command()
@click.argument('filename')
@click.option('-t', '--tamaki-time', default=15)
@click.option('-T', '--max-tw', default=32)
@click.option('-s', '--slice-step', default=None, type=int)
@click.option('-C', '--cost-type', default='length')
def opt_file(filename, tamaki_time, max_tw, slice_step, cost_type):
    tn = qop.TensorNet.QtreeTensorNet.from_qsim_file(filename)
    fopt = qop.Optimizer.TamakiTrimSlicing()
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
@click.option('-G','--graph-type', default='random_regular')
def generate_qaoa_ansatz_circuit(seed, degree, nodes, p, graph_type):
    np.random.seed(seed)
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
    composer = QtreeQAOAComposer(G, beta=beta, gamma=gamma)
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
    composer = QtreeQAOAComposer(G, beta=beta, gamma=gamma)
    composer.energy_expectation_lightcone(edge)
    txt = qtree.operators.circuit_to_text([composer.circuit], nodes)
    print(txt)

@cli.command()
@click.option('-s','--seed', default=42)
@click.option('-d','--degree', default=3)
@click.option('-n','--nodes', default=10)
@click.option('-p','--p', default=1)
@click.option('-G','--graph-type', default='random_regular')
@click.option('-T','--max-time', default=0, help='Max time for every evaluation')
@click.option('-O','--ordering-algo', default='greedy', help='Algorithm for elimination order')
def qaoa_energy_tw(nodes, seed, degree, p, graph_type, max_time, ordering_algo):
    np.random.seed(seed)
    if graph_type=='random_regular':
        G = nx.random_regular_graph(degree, nodes)
    elif graph_type=='erdos_renyi':
        G = nx.erdos_renyi_graph(nodes, degree/(nodes-1))
    elif graph_type=='erdos_renyi_core':
        G = nx.erdos_renyi_graph(nodes, degree/(nodes-1))
        print('degrees', list(G.degree))
        G = nx.algorithms.core.k_core(G, k=degree)
        print('nodes', G.number_of_nodes())
    else:
        raise Exception('Unsupported graph type')
    gamma, beta = [0]*p, [0]*p

    def get_tw(circ):

        tn = QtreeTensorNet.from_qtree_gates(circ)

        if ordering_algo=='greedy':
            opt = OrderingOptimizer()
        elif ordering_algo=='tamaki':
            opt = TamakiOptimizer(wait_time=45)
        elif ordering_algo=='without':
            opt = WithoutOptimizer()
        else:
            raise ValueError("Ordering algorithm not supported")
        peo, tn = opt.optimize(tn)
        treewidth = opt.treewidth
        return treewidth

    twidths = []
    if max_time:
        start = time.time()
    else:
        start = np.inf
    for edge in tqdm(G.edges()):
        composer = QtreeQAOAComposer(G, beta=beta, gamma=gamma)
        composer.energy_expectation_lightcone(edge)
        tw = get_tw(composer.circuit)
        twidths.append(tw)
        if time.time() - start > max_time:
            break
    print(f'med={np.median(twidths)} mean={round(np.mean(twidths), 2)} max={np.max(twidths)}')

cli()
