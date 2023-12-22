import qtensor
import qtree
import networkx as nx
import numpy as np

# -- QAOA generic parser

def parse_qaoa_composer(data):
    terms = data["terms"]
    gamma = np.array(data["gamma"])/np.pi/2
    beta = np.array(data["beta"])/np.pi
    N = len(set(sum([t[1] for t in terms], [])))
    G = nx.Graph()
    for factor, term in terms:
        G.add_edge(*term)
    composer = qtensor.DefaultQAOAComposer(G, gamma=gamma, beta=beta)
    return composer
# --


def read_preps(prep_f):
    import pickle
    return pickle.load(prep_f.f)

def write_preps(peo, prep_f):
    import pickle
    pickle.dump(peo, open(prep_f, 'wb'))

def write_json(data, out_file):
    import json
    with open(out_file, 'w') as f:
        json.dump(data, f)
        # This newline plays nice when cat-ing multiple files
        f.write('\n')

def preprocess_circ(circ, S, O, M, after_slice):
    tn = qtensor.optimisation.QtreeTensorNet.from_qtree_gates(circ)
    opt = qtensor.toolbox.get_ordering_algo(O)
    if S:
        # ignore argument type mismatch for pyright -- opt can be `Optimizer`
        # pyright: reportGeneralTypeIssues=false
        opt = qtensor.optimisation.TreeTrimSplitter(
            tw_bias=0, max_tw=M, base_ordering=opt,
            peo_after_slice_strategy=after_slice
        )
        
        peo, par_vars, _ = opt.optimize(tn)
        # --dbg
        graph = tn.get_line_graph()
        ignore_vars = tn.bra_vars + tn.ket_vars
        for pv in par_vars:
            graph.remove_node(int(pv))
        components = list(nx.connected_components(graph))
        print(f"Sliced graph # nodes: {graph.number_of_nodes()} and #components: {len(components)} with sizes {[len(c) for c in components]}")
        print(f"peo size without par_vars and ignore_vars: {len(peo) - len(par_vars) - len(ignore_vars)}")

        print()
        # --
    else:
        peo, _ = opt.optimize(tn)
        par_vars = []
    #print("W", opt.treewidth)
    return (peo, par_vars, tn), opt.treewidth

def preprocess(composer, out_file, O='greedy', S=None, M=30, after_slice='run-again'):
    """
    Arguments:
        composer: input file
        out_file: output file
        O: ordering algorithm 
        S: slicing algorithm 
        M: Memory limit for slicing 
    """
    import copy
    G = composer.graph
    prep_data = []
    for edge in G.edges:
        c_copy = copy.deepcopy(composer)
        c_copy.energy_expectation_lightcone(edge)
        e_prep, treewidth = preprocess_circ(c_copy.circuit, S, O, M, after_slice)
        if treewidth>25:
            prep_data.append(e_prep)
    write_preps(prep_data, out_file)
    print(f"Wrote {len(prep_data)} preparations of lightcones")
    return prep_data

def estimate(in_file, out_file, C=100, M=30, F=1e12, T=1e9, **kwargs):
    """
    Arguments:
        in_file: file with preprocessed data
        out_file: file to write the results to
        C: Compression ratio
        M: Memory limit in log2(b/16)
        F: assumed FLOPS 
        T: Throughput of compression
    """
    from qtensor.compression.cost_estimation import compressed_contraction_cost, Cost
    from dataclasses import asdict
    import json
    prep_data = read_preps(in_file)
    peo, par_vars, tn = prep_data

    tn.slice({i: slice(0, 1) for i in par_vars})
    peo = peo[:len(peo) - len(par_vars)]
    costs: list[Cost] = compressed_contraction_cost(tn, peo, mem_limit=M, compression_ratio=C)
    totals: Cost = sum(costs[1:], costs[0])
    time = totals.time(F, T, T, M)
    C = asdict(totals)
    C['time'] = time*2**len(par_vars)
    C['slices'] = 2**len(par_vars)
    print("C", C)
    out_file += ".json"
    write_json(C, out_file)
    return out_file

def simulate(in_file, out_file,
             backend='einsum',
             compress=None,
             M=29,
             r2r_error=1e-3, r2r_threshold=1e-3,
             **kwargs):
    import cupy
    prep_data = read_preps(in_file)
    cupy.cuda.profiler.start()

    C = dict(
        time=0,
        elapsed=0,
        memory=0,
        memory_history=[],
        nvmemory=0,
        result = dict(Re=0, Im=0),
        compression=dict(compress=[], decompress=[])
    )

    for prep_lightcone in prep_data[:5]:
        print(prep_lightcone)
        r = simulate_preps_lightcone(prep_lightcone, backend, compress, M,
                                              r2r_error,
                                              r2r_threshold,**kwargs)
        C['time'] += r['time']
        C['elapsed'] += r['elapsed']
        C['memory'] = max(C['memory'], r['memory'])
        C['nvmemory'] = max(C['nvmemory'], r['nvmemory'])
        C['memory_history'] += r['memory_history']
        C['result']['Re'] += r['result']['Re']
        C['result']['Im'] += r['result']['Im']
        if r.get('compression'):
            C['compression']['compress'] += r['compression']['compress']
            C['compression']['decompress'] += r['compression']['decompress']

    out_file += ".json"
    write_json(C, out_file)
    return out_file
    cupy.cuda.profiler.stop()

def simulate_preps_lightcone(prep_data,
             backend='einsum',
             compress=None,
             M=29,
             r2r_error=1e-3, r2r_threshold=1e-3,
             **kwargs):
    """
    Args:
        in_file: file with preprocessed data
        out_file: file to write the results to
        backend: backend to use
        compress: compression algorithm
        M: memory threshold for compression
        r2r_error: relative error for compression
        r2r_threshold: relative threshold for compression
    """
    import time
    from qtensor.contraction_algos import bucket_elimination
    from qtensor.compression.Compressor import CUSZCompressor, CUSZXCompressor, TorchCompressor, NEWSZCompressor
    #from qtensor.compression.Compressor import WriteToDiskCompressor
    import cupy
    peo, par_vars, tn = prep_data
    
    backend = qtensor.contraction_backends.get_backend(backend)
    if compress is not None:
        if compress == 'szx':
            print(f"{r2r_error=} {r2r_threshold=}")
            compressor = CUSZXCompressor(r2r_error=r2r_error, r2r_threshold=r2r_threshold)
            compressor = qtensor.compression.ProfileCompressor(compressor)
        elif compress == 'cusz':
            print(f"{r2r_error=} {r2r_threshold=}")
            compressor = CUSZCompressor(r2r_error=r2r_error, r2r_threshold=r2r_threshold)
            compressor = qtensor.compression.ProfileCompressor(compressor)
        elif compress == 'torch':
            print(f"{r2r_error=} {r2r_threshold=}")
            compressor = TorchCompressor(r2r_error=r2r_error, r2r_threshold=r2r_threshold)
            compressor = qtensor.compression.ProfileCompressor(compressor)
        elif compress == 'newsz':
            print(f"{r2r_error=} {r2r_threshold=}")
            compressor = NEWSZCompressor(r2r_error=r2r_error, r2r_threshold=r2r_threshold)
            compressor = qtensor.compression.ProfileCompressor(compressor)
        elif compress == 'disk':
            compressor = WriteToDiskCompressor(f'/grand/QTensor/compression/data/tensors_compressed_M{M}/')
            compressor = qtensor.compression.ProfileCompressor(compressor)
        else:
            raise ValueError(f"Unknown compression algorithm: {compress}")
        backend = qtensor.contraction_backends.CompressionBackend(backend, compressor, M)
        from qtensor.contraction_backends.performance_measurement_decorator import MemProfBackend
        backend = MemProfBackend(backend)

    relabelid = {}
    for tensor in tn.tensors:
        for i in tensor.indices:
            relabelid[int(i)] = i

    slice_ext = {relabelid[int(i)]: 0 for i in par_vars}

    if len(par_vars) > 0:
        print("Parvars", par_vars)
        print(f"Detected {len(par_vars)} slice variables")
    sim = qtensor.QtreeSimulator(backend=backend)
    sim.tn = tn
    sim.tn.backend = backend
    sim.peo = peo
    sim._slice_relabel_buckets(slice_ext)
    buckets = sim.tn.buckets
    # --dbg
    #ignore_vars  = sim.tn.bra_vars + sim.tn.ket_vars 
    #graph = qtree.graph_model.importers.buckets2graph(buckets, ignore_vars)
    #graph, label_dict = qtree.graph_model.relabel_graph_nodes(
        #graph, dict(zip(graph.nodes, np.array(list(graph.nodes)) - 127*2))
    #) 
    #import networkx as nx
    #components = list(nx.connected_components(graph))
    #print(f"Sliced graph # nodes: {graph.number_of_nodes()} and #components: {len(components)} with sizes {[len(c) for c in components]}")
    #print(f"peo size without par_vars and ignore_vars: {len(peo) - len(ignore_vars)}")
    # --

    start = time.time()
    for i in range(2**0):
        print(f"P {i}", end='', flush=True)
        bcopy = [b[:] for b in buckets]
        res = bucket_elimination(
            bcopy, backend,
            n_var_nosum=len(tn.free_vars)
        )
        del bcopy
        print("Result", res.data.flatten()[0])
        #time.sleep(0.5)
    sim_result = backend.get_result_data(res).flatten()[0]
    print("Simulation result:", sim_result)
    end = time.time()
    print("Elapsed", end - start)
    C = {'time': 2**len(par_vars)*(end - start)}
    C['elapsed'] = (end - start)
    C['memory'] = backend.max_mem
    C['memory_history'] = backend.mem_history
    C['nvmemory'] = backend.nvsmi_max_mem
    C['result'] = {
        "Re": np.real(sim_result).tolist(),
        "Im": np.imag(sim_result).tolist()
    }
    if compress is not None:
        if isinstance(compressor, qtensor.compression.ProfileCompressor):
            C['compression'] = compressor.get_profile_data_json()
    return C
