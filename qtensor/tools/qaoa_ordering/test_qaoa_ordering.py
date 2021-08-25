import qtensor
import qtree
import networkx as nx


def test_column_verts():
    from qtensor.tools.qaoa_ordering import circ2gvars
    g = nx.Graph()
    g.add_edge(0, 1)
    p = 2
    comp = qtensor.DefaultQAOAComposer(g, gamma=[1]*p, beta=[2]*p)
    comp.energy_expectation_lightcone((0, 1))
    cverts = circ2gvars(g.number_of_nodes(), [comp.circuit])
    print('cverts', cverts)
    assert len(cverts.keys()) == g.number_of_nodes()
    assert len(cverts[0]) == 2*p + 2 + 1

    tn = qtensor.optimisation.QtreeTensorNet.from_qtree_gates(comp.circuit)
    lg = tn.get_line_graph()
    verts = set(lg.nodes)
    cvert_merge = sum(cverts.values(), [])
    cvert_merge = [int(x) for x in cvert_merge]
    assert len(cvert_merge) == len(verts)
    assert set(cvert_merge) == verts


def test_qaoa_ordering():
    from qtensor.tools.qaoa_ordering import get_qaoa_exp_ordering
    p = 3
    N = 50
    g = nx.path_graph(N)  
    comp = qtensor.DefaultQAOAComposer(g, gamma=[1]*p, beta=[2]*p)
    comp.energy_expectation_lightcone((0, 1))
    order = get_qaoa_exp_ordering(comp.circuit, algo='greedy')
    tn = qtensor.optimisation.QtreeTensorNet.from_qtree_gates(comp.circuit)
    lg = tn.get_line_graph()
    width = qtree.graph_model.get_treewidth_from_peo(lg, peo=order)
    
    assert width == 2*p