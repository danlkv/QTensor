from qtensor.tools.benchmarking import qc

def test_avail_brist():
    params = qc.get_avail_bris_params(4)

def test_avail_rect():
    params = qc.get_avail_rect_params(4)
    print(params)


def test_read_brist():
    n, circ = qc.get_bris_circuit(diag=4, layers=24, seed=2)
    assert isinstance(circ, list)
    assert n == 12

    n, circ = qc.get_bris_circuit(diag=4, layers=40, seed=2)
    assert isinstance(circ, list)
    assert n == 12


def test_read_rect():
    n, circ = qc.get_rect_circuit(4, layers=24, seed=2)
    assert isinstance(circ, list)
    assert n == 16

    n, circ = qc.get_rect_circuit(4, layers=40, seed=2)
    assert isinstance(circ, list)
    assert n == 16
