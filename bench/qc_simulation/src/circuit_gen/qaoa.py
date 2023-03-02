import networkx
import numpy as np

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

def generate_maxcut(out_file, N, p, d, graph_type='random', seed=None):
    """
    Generate a random regular maxcut problem

    Args:
        out_file (str): Path to output file
        N (int): Number of nodes
        p (int): Number of layers
        d (int): Random regular graph degree 

    Returns:
        str: Path to output file
    """
    G = generate_graph(N, d, graph_type)
    terms = []
    for u, v in G.edges:
        terms.append((1, (u, v)))
    gamma = np.random.uniform(0, 2 * np.pi, p)
    beta = np.random.uniform(0, np.pi, p)
    pb = {"terms": terms, "gamma": gamma.tolist(), "beta": beta.tolist()}

    return save_terms_format(out_file, pb)
