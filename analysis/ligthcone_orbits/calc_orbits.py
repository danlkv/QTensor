import qtensor
from qtensor.tools import lightcone_orbits
from qtensor.tools.mpi.mpi_map import MPIParallel
import networkx as nx


def main():
    N = int(1e4)
    d = 3
    p = 3
    G = nx.random_regular_graph(d, N, seed=42)

    par = MPIParallel()

    print(f'Calculating orbits of N={N} graph. d={d} p={p}')
    try:
        orbits = lightcone_orbits.get_edge_orbits_lightcones(G, p, nprocs=par)
    except TypeError as e:
        # debt: should handle mpi ranks properly via MPIParallel object
        # hope that this happened for mpi rank 1
        return
    freq_dict = { edges[0]: len(edges) for edges in orbits.values() }
    freq_report = str(freq_dict)
    if len(freq_report)>10000:
        freq_report = freq_report[:10000] + "[truncated]"
    print('freq report', freq_report)
    print('Number of orbits', len(freq_dict))

if __name__=='__main__':
    main()
