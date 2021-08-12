import qtensor
from qtensor.optimisation.kahypar_ordering import generate_TN

def test_dual_hg():
    hg = {1: [1, 2, 3], 2: [1, 4], 3: []}
    dual_expect = {1: [1, 2], 2: [1], 3: [1], 4: [2]}
    dual = generate_TN.dual_hg(hg)
    assert dual==dual_expect

def test_tn():
    import networkx as nx
    N = 5
    g = nx.path_graph(N)
    """
    Resulting TN (hypergraph):
        --- input ---

        |    |    |
        M    M    M
        |    |    |
        H    H    H
        |\  /|\  /|
         -ZZ- -ZZ- -
        |/  \|/  \|
        U    U    U
        |    |    |
        M    M    M
        |    |    |

        --- output ---

    """
    dangling_cnt = 2*N
    vert_cnt = 4*N + (N-1)
    edge_cnt = 5*N

    comp = qtensor.DefaultQAOAComposer(g, gamma=[1], beta=[2])
    comp.ansatz_state()
    circ = comp.circuit
    tn_ = qtensor.optimisation.QtreeTensorNet.from_qtree_gates(circ)
    print('tensors', tn_.tensors)

    tn = generate_TN.circ2tn(circ)
    print(tn)
    dangling = [item for item in tn.values() if len(item)==1]

    assert len(dangling) == dangling_cnt
    verts = set(sum(tn.values(), []))

    assert len(verts) == vert_cnt

    assert len(tn) == edge_cnt
