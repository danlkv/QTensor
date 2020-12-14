import qtensor
import numpy as np


def test_torch_sim():
    import torch
    p = 3
    gamma, beta = torch.tensor([0.1]*p), torch.tensor([0.2]*p)
    G = qtensor.toolbox.random_graph(nodes=10, degree=3)
    composer = qtensor.TorchQAOAComposer(G, gamma=gamma, beta=beta)
    composer.ansatz_state()
    circ = composer.circuit

    sim = qtensor.QtreeSimulator(backend=qtensor.contraction_backends.TorchBackend())
    restr = sim.simulate(circ)
    assert isinstance(restr, torch.Tensor)

    composer = qtensor.DefaultQAOAComposer(G, gamma=gamma, beta=beta)
    composer.ansatz_state()
    circ = composer.circuit

    sim = qtensor.QtreeSimulator()
    resnp = sim.simulate(circ)
    assert np.allclose(resnp, restr)

def test_torch_composer__smoke():
    import torch
    p = 3
    gamma, beta = torch.tensor([0.1]*p), torch.tensor([0.2]*p)
    G = qtensor.toolbox.random_graph(nodes=10, degree=3)
    composer = qtensor.TorchQAOAComposer(G, gamma=gamma, beta=beta)
    composer.ansatz_state()
    circ = composer.circuit
    assert len(circ)

    composer.builder.reset()
    composer.energy_expectation_lightcone(list(G.edges)[0])
    assert len(composer.circuit)

def test_torch_gates():
    import torch
    def compare_gates(a, b):
        assert np.allclose(a.gen_tensor(), b.gen_tensor())
    alphatr = torch.tensor(0.19)
    alphanp = 0.19
    ftr = qtensor.OpFactory.TorchFactory
    fnp = qtensor.OpFactory.QtreeFactory

    compare_gates(
        ftr.ZZ(0, 1, alpha=alphatr),
        fnp.ZZ(0, 1, alpha=alphanp)
    )

    compare_gates(
        ftr.ZPhase(0, alpha=alphatr),
        fnp.ZPhase(0, alpha=alphanp)
    )

    compare_gates(
        ftr.XPhase(0, alpha=alphatr),
        fnp.XPhase(0, alpha=alphanp)
    )

