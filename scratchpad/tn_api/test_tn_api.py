import random
import numpy as np
from tn import TensorNetwork
from functools import reduce

def test_add_numpy_array():
    a = TensorNetwork()
    t = np.random.randn(2, 2)
    a.add(t)
    b = TensorNetwork()
    b.add(a)
    assert b == a


def test_composition():
    """
    tensor network adding is associative
    """
    tns = [TensorNetwork.new_random_cpu(2, 3, 4) for _ in range(5)]
    stack = TensorNetwork()
    # (((0A)B)C)D
    for tn in tns:
        stack.add(tn)
    # A(B(CD))
    for i in range(len(tns)):
        l = tns[len(tns)-2-i]
        r = tns[len(tns)-1-i]
        l.add(r)

    assert stack == tns[0]

def test_edges_consistent_ports():
    tns = [TensorNetwork.new_random_cpu(2, 3, 4) for _ in range(5)]
    tn = TensorNetwork()
    # (((0A)B)C)D
    for t in tns:
        tn.add(t)

    port_data = {}
    for e in tn._edges:
        for p in e:
            port_data[p.tensor_ref] = port_data.get(p.tensor_ref, [])
            port_data[p.tensor_ref].append(p.ix)
    for i, t in enumerate(tn._tensors):
        assert len(t.shape) == len(port_data[i])



if __name__=="__main__":
    test_edges_consistent_ports()
    test_composition()
