import sys
import click
import qtree.operators as ops
from qensor.FeynmanSimulator import FeynmanSimulator
import qensor.optimisation as qop

@click.group()
def cli():
    pass

@cli.command()
@click.argument('filename')
def sim_file(filename):
    n_qubits, circuit = ops.read_circuit_file(filename)
    sim = FeynmanSimulator()
    circuit = sum(circuit, [])
    result = sim.simulate(circuit, batch_vars=4, tw_bias=0)
    print(result)

@cli.command()
@click.argument('filename')
@click.option('-t', '--tamaki-time', default=15)
@click.option('-T', '--max-tw', default=32)
@click.option('-s', '--slice-step', default=None, type=int)
def opt_file(filename, tamaki_time, max_tw, slice_step):
    tn = qop.TensorNet.QtreeTensorNet.from_qsim_file(filename)
    fopt = qop.Optimizer.TreeTrimSplitter()
    fopt.max_tw = max_tw
    fopt.par_var_step = slice_step
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


cli()
