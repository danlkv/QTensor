import qtensor

def test_torch_gates():
    import torch
    alpha = torch.tensor(0.19)
    ZZ = qtensor.OpFactory.TorchFactory.ZZ(0, 1, alpha=alpha)
    tensor = ZZ.gen_tensor()

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (2,2)

