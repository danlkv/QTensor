import qtensor
from qtensor.contraction_backends import TorchBackend


def test_torch_get_sliced_smoke():
    backend = TorchBackend()
    test_buckets = []
    buckets = backend.get_sliced_buckets(test_buckets, {}, {})
    assert buckets == []

