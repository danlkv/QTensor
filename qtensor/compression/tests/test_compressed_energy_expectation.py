import qtensor
import numpy as np
from qtensor.compression import CUSZPCompressor
import qtensor.tests

def test_compress_energy_expect():
    G, gamma, beta = qtensor.tests.get_test_problem(n=10, p=2, type='random')
    edge = list(G.edges())[0]
    composer = qtensor.QtreeQAOAComposer(G, gamma=gamma, beta=beta)
    composer.energy_expectation_lightcone(edge)
    circuit = composer.circuit
    base_backend = qtensor.contraction_backends.get_backend('cupy')
    compressor = CUSZPCompressor(r2r_error=1e-4, r2r_threshold=1e-4)
    backend = qtensor.contraction_backends.CompressionBackend(base_backend, compressor, max_tw=6)
    sim = qtensor.QtreeSimulator(backend=backend)
    res = sim.simulate(circuit)
    sim_exact = qtensor.QtreeSimulator(backend=base_backend)
    ref = sim_exact.simulate(circuit)
    print(f'exact: {ref}, compressed: {res}')
    assert np.allclose(res, ref, atol=1e-4, rtol=0.05)

if __name__ == '__main__':
    test_energy_expect()
    print('test passed!')
