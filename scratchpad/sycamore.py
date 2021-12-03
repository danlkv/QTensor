import qtensor
import qtree
from qtree.graph_model.exporters import generate_gr_file
from qtree.graph_model.base import get_simple_graph, relabel_graph_nodes
from qtensor.tools.benchmarking import qc

if __name__=="__main__":
    n, circ = qc.get_syc_circ(53, 20)
    gr, data_dict, bra, ket = qtree.graph_model.circ2graph(n, circ)
    graph = get_simple_graph(gr)
    graph, initial_to_conseq = relabel_graph_nodes(
        graph,
        dict(zip(graph.nodes, range(1, graph.number_of_nodes()+1))))
    data = generate_gr_file(graph)
    print(data)
