# it is meant to init ipython env
import utils as u
import utils_qaoa as qu
import qtree as qt

qc, N = qu.get_test_qaoa(53, 1, seed=42, type='randomreg',degree=3)

bck, data, bra, ket = qt.optimizer.circ2buckets(N, qc)

G = qt.graph_model.circ2graph(N, qc)
G2 = qt.graph_model.buckets2graph(bck)

p1, n1 = u.get_locale_peo(G, u.n_neighbors)
p2, n2 = u.get_locale_peo(G2, u.n_neighbors)

