import networkx as nx
from qtensor.tools.lazy_import import gurobipy as gb
from typing import Tuple

def solve_maxcut(G: nx.Graph) -> Tuple[float, list]:
    """

    Some piece of info:
    https://twitter.com/nkrislock/status/432605891322318849
    """

    p = gb.Model()
    p.setParam('PreQLinearize', 1)

    vdict = {}
    for n in G.nodes:
        vdict[n] = p.addVar(name='v_'+str(n), vtype=gb.GRB.BINARY)
    scaled_v = {v:(2*x - 1) for v, x in vdict.items()}
    C_i = [vdict[i] + vdict[j] - 2*vdict[i]*vdict[j] for i, j in G.edges]
    p.setObjective(sum(C_i), gb.GRB.MAXIMIZE)
    p.optimize()
    reverse_map = {v:k for k, v in vdict.items()}
    return p.objVal, [int(vdict[n].x) for n in G.nodes]

