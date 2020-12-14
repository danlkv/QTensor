import qtensor

def test_torch_sim():
    import torch
    p = 3
    gamma, beta = torch.tensor([0.1]*p), torch.tensor([0.2]*p)
    G = qtensor.toolbox.random_graph(nodes=10, degree=3)
    composer = qtensor.TorchQAOAComposer(G, gamma=gamma, beta=beta)
    composer.ansatz_state()
    circ = composer.circuit

    sim = qtensor.QtreeSimulator(backend=qtensor.contraction_backends.TorchBackend())
    res = sim.simulate(circ)
    assert isinstance(res, torch.Tensor)

def test_torch_composer():
    import torch
    p = 3
    gamma, beta = torch.tensor([0.1]*p), torch.tensor([0.2]*p)
    G = qtensor.toolbox.random_graph(nodes=10, degree=3)
    composer = qtensor.TorchQAOAComposer(G, gamma=gamma, beta=beta)
    composer.ansatz_state()
    circ = composer.circuit
    assert len(circ)


def test_torch_gates():
    import torch
    alpha = torch.tensor(0.19)
    ZZ = qtensor.OpFactory.TorchFactory.ZZ(0, 1, alpha=alpha)
    tensor = ZZ.gen_tensor()

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (2,2)

