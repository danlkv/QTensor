import qtensor
import qtree
import networkx as nx

def get_gb(p):
    return [0.1]*p, [0.2]*p

def main():
    sim = qtensor.QAOAQtreeSimulator(qtensor.DefaultQAOAComposer)
    gamma, beta =  get_gb(2)
    comp = qtensor.DefaultQAOAComposer(nx.random_regular_graph(3, 16), gamma=gamma, beta=beta)
    comp.ansatz_state()
    sim.prepare_buckets(comp.circuit)
    buckets = sim.buckets
    # -- Do some contraction to populate buckets
    backend = qtensor.contraction_backends.NumpyBackend()
    print('>Starting contract. Bucktes:', buckets)
    qtree.optimizer.bucket_elimination(
        buckets, backend.process_bucket,
        n_var_nosum = 5
    )
    i = 2
    print('>Stopped contract. Buckets:', buckets)
    print(f'Bucket [{i}]:', buckets[i])



if __name__=="__main__":
    main()
