import sys
sys.path.append('.')
from test_circuits import gen_qaoa_maxcut_circuit
import qtensor
import qtree
import numpy as np
import pandas as pd
import pyrofiler

from qtensor import QtreeQAOAComposer
from qtensor import QtreeSimulator
from qtensor import toolbox
from qtensor.contraction_backends import get_backend, PerfBackend

def bucket_contraction_report(tn, buckets, backend,
                              bucket_elimination=qtree.optimizer.bucket_elimination
                             ):
    """
    Returns:
        str: string summary of report
        report_table: table of metrics for each bucket contraction step
        profile_results: raw bucket data: dict mapping `str: (indices, time)`
    """

    perf_backend = PerfBackend(print=False, num_lines=20)
    perf_backend.backend = backend
    result = bucket_elimination(
        buckets, perf_backend.process_bucket,
        n_var_nosum=len(tn.free_vars)
    )
    perf_backend.get_result_data(result).flatten()
    # compute report_table
    rep_txt = perf_backend.gen_report(show=False)
    return rep_txt, perf_backend.report_table, perf_backend._profile_results

def get_buckets_tn(circ, backend, ordering_algo:str, batch_vars=0, seed=10):
    np.random.seed(seed)
    tn = qtensor.optimisation.QtreeTensorNet.from_qtree_gates(circ)
    opt = toolbox.get_ordering_algo(ordering_algo)
    sim = QtreeSimulator(optimizer=opt, backend=backend)
    sim.prepare_buckets(circ, batch_vars=batch_vars)
    return sim.buckets, tn

def gen_circuit_simulation_report(backend, circ):
    timing = pyrofiler.timing
    with timing(callback=lambda x: None) as gen:
        buckets, tn = get_buckets_tn(circ, backend, 'greedy', batch_vars=0)

    text, report_table, profile_results = bucket_contraction_report(tn, buckets, backend)

    return report_table

'''
Function: Generate a collection of above report, and process them into final usable form
I/O: ... -> processed data is a dict, directly usable by json
'''
def collect_process_be_pt_report(repeat: int, backend, circ):
    tables = []

    for _ in range(repeat):
        table = gen_circuit_simulation_report(backend, circ)
        tables.append(table)
    report = pd.concat(tables, keys=range(repeat), names=['repeat', 'step'])
    return report


def mean_mmax(x: list):
    mx, mn = max(x), min(x)
    x.remove(mx)
    x.remove(mn)
    return np.mean(x)

def main():
    N = 22
    p = 3
    backend_name = 'torch_cpu'
    backend = get_backend(backend_name)
    circ = gen_qaoa_maxcut_circuit(N, p)
    report = collect_process_be_pt_report(15, backend, circ)

    stats = report[["time"]].groupby('step').agg(['mean', 'min', 'max', 'std'])
    stats = pd.concat([
        stats,
        report[['result_size']].groupby('step').agg(['mean'])
    ], axis=1)
    stats.sort_values(by=[('time', 'mean')], ascending=False, inplace=True)
    top_K = 12
    print(f"Top {top_K} steps by time:")
    print(stats.head(top_K))
    print(stats[('time', 'mean')].sum())

if __name__=="__main__":
    main()


