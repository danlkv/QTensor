import sys
import lzma
import networkx as nx

ENCODING = sys.getdefaultencoding()


def generate_gr_file(graph, filename="", compressed=False):
    """
    Generate a gr format input file for the graph.
    This function ALWAYS expects a simple Graph without self loops

    Parameters
    ----------
    graph : networkx.Graph
           Undirected graphical model
    filename : str
           Output file name. If not specified,
           the contents are returned as a string
    compressed : bool
           if output data should be compressed
    Returns
    -------
    data: str or bytes or None
    """
    v = graph.number_of_nodes()
    e = graph.number_of_edges()

    data = "c a configuration of the graph\n"
    data += f"p tw {v} {e}\n"

    for edge in graph.edges():
        u, v = edge
        # print only if this is not a self-loop
        # Node numbering in this format is 1-based
        if u != v:
            data += '{} {}\n'.format(int(u), int(v))

    if compressed:
        data = lzma.compress(data.encode(ENCODING))

    if len(filename) > 0:
        with open(filename, 'w+') as fp:
            fp.write(data)
        return None
    else:
        return data


def generate_cnf_file(graph, filename="", compressed=False):
    """
    Generate QuickBB input file for the graph.
    This function ALWAYS expects a simple Graph (not MultiGraph)
    without self loops
    because QuickBB does not understand these situations.

    Parameters
    ----------
    graph : networkx.Graph
           Undirected graphical model
    filename : str
           Output file name. If not specified,
           the contents are returned as a string
    compressed : bool
           if output data should be compressed
    Returns
    -------
    data: str or bytes or None
    """
    v = graph.number_of_nodes()
    e = graph.number_of_edges()
    data = "c a configuration of -qtree simulator\n"
    data += f"p cnf {v} {e}\n"

    # Convert possible MultiGraph to Graph (avoid repeated edges)
    for edge in graph.edges():
        u, v = edge
        # print only if this is not a self-loop
        # Node numbering in this format is 1-based
        if u != v:
            data += '{} {} 0\n'.format(int(u), int(v))

    if compressed:
        data = lzma.compress(data.encode(ENCODING))

    if len(filename) > 0:
        with open(filename, 'w+') as fp:
            fp.write(data)
        return None
    else:
        return data

