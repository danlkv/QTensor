import networkx
import numpy as np
from qtensor.tools import BETHE_QAOA_VALUES

def generate_ibm_connectivity(arch):
    """
    Generate a connectivity graph from an IBM architecture

    Args:
        arch (str): one of ["eagle", "falcon"]
    """
    supported_archs = ["eagle", "falcon"]
    if arch not in supported_archs:
        raise ValueError("Architecture {} not supported".format(arch))

    def coupling_map_from_provider(p_class):
        p = p_class()
        graph = p.coupling_map.graph.to_undirected()
        elist = list(graph.edge_list())
        G = networkx.from_edgelist(elist)
        return G

    if arch == "eagle":
        # IBM quantum volume 64
        from qiskit.providers.fake_provider import FakeWashingtonV2
        return coupling_map_from_provider(FakeWashingtonV2)
    if arch == "eagle":
        # IBM quantum volume 64
        from qiskit.providers.fake_provider import FakeCairoV2
        return coupling_map_from_provider(FakeCairoV2)
    else:
        raise ValueError("IBM architecture {} not supported".format(arch))

def save_terms_format(file, terms):
    """
    Save terms in a format that can be read by the qtensor simulator Takes a
    list of terms in format `(coeff, [qubits])` and saves it to a file
    """
    import json
    filename = file + '.jsonterms'
    with open(filename, "w") as f:
        json.dump(terms, f)
    return filename

def generate_graph(n, d, type="random"):
    if type == "random":
        return networkx.random_regular_graph(d, n)
    elif type[:4] == "ibm_":
        arch = type[4:]
        return generate_ibm_connectivity(arch)
    else:
        raise ValueError("Unknown graph type {}".format(type))

def generate_maxcut(out_file, N, p, d, graph_type='random', seed=None, parameters='random'):
    """
    Generate a random regular maxcut problem

    Args:
        out_file (str): Path to output file
        N (int): Number of nodes
        p (int): Number of layers
        d (int): Random regular graph degree 
        parameters (str): One of ["random", "fixed_angles"]

    Returns:
        str: Path to output file
    """
    G: networkx.Graph = generate_graph(N, d, graph_type)
    terms = []
    for u, v in G.edges:
        terms.append((1, (u, v)))
    if parameters == "random":
        gamma = np.random.uniform(0, 2 * np.pi, p)
        beta = np.random.uniform(0, np.pi, p)
    elif parameters == "fixed_angles":
        gammabeta = np.array(BETHE_QAOA_VALUES[str(p)]['angles'])
        gamma, beta = gammabeta[:p]*2, gammabeta[p:]
    else:
        raise ValueError("Unknown parameters type {}. Use one of ['random', 'fixed_angles']".format(parameters))
    pb = {"terms": terms, "gamma": gamma.tolist(), "beta": beta.tolist()}

    return save_terms_format(out_file, pb)
