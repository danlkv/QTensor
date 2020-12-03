#import networkit as nk
from qtensor.tools.lazy_import import networkit as nk
from itertools import combinations
import numpy as np

def ordering_opt(G, max_width=np.inf, max_steps=np.inf):
    nk_to_nx_labels = {k:v for k, v in enumerate(G.nodes())}
    G = nk.nxadapter.nx2nk(G)
    G = nk.Graph(G)
    G.removeSelfLoops()
    ordering = []
    widths = []
    nodes = np.array(list(G.iterNodes()))
    degs = np.array(list(map(G.degree, nodes)))
    nghs = nodes

    while G.numberOfNodes():
        loc_deg = list(map(G.degree, nghs))
        degs[nghs] = loc_deg
        best_ix = degs.argmin()
        best = degs[best_ix]
        best_node = nodes[best_ix]
        ordering.append(best)
        widths.append(best)
     #  if best>max_width:
     #      return ordering 
     #  if len(ordering)>max_steps:
     #      return ordering

        nghs = list(G.iterNeighbors(best_node))
        #print('best', best_node, 'bestix', best_ix ,'lenngs', len(ngs), ngs)
        G.removeNode(best_node)
        degs[best_ix] = 1000_000
        #print('lennedges', len(new_edges))
        #print('lennedges', len(new_edges))
        if len(nghs)>1:
            new_edges = list(combinations(nghs, 2))
            existing = []
            for u in nghs:
                existing += [(u, v) for v in G.iterNeighbors(u)]

            new_edges = set(new_edges) - set(existing)#[e for e in new_edges if e not in existing]
            if len(new_edges)>0:
                [G.addEdge(*e) for e in new_edges]
            #print(f'added {len(new_edges)} edges')

      # for u in nghs:
      #     existing = list(G.iterNeighbors(u))
      #     for v in nghs:
      #         if v not in existing:
      #             if u!=v:
      #                 G.addEdge(u, v)
      #     #print(f'added {len(new_edges)} edges')
        if best>100:
            raise Exception('wtf')
    ordering = [nk_to_nx_labels[i] for i in ordering]
    return ordering, widths
