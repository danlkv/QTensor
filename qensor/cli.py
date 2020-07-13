import sys
import click
import qtree.operators as ops
from qensor.FeynmanSimulator import FeynmanSimulator

@click.command()
@click.argument('filename')
def sim_file(filename):
    n_qubits, circuit = ops.read_circuit_file(filename)
    sim = FeynmanSimulator()
    circuit = sum(circuit, [])
    result = sim.simulate(circuit, batch_vars=4, tw_bias=0)
    print(result)

sim_file()
