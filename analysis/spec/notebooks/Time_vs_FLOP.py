# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Qtensor-simulation-times" data-toc-modified-id="Qtensor-simulation-times-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Qtensor simulation times</a></span><ul class="toc-item"><li><span><a href="#Get-cost-estimation" data-toc-modified-id="Get-cost-estimation-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Get cost estimation</a></span></li><li><span><a href="#Simulate-the-circuit" data-toc-modified-id="Simulate-the-circuit-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Simulate the circuit</a></span></li><li><span><a href="#Compare-estimation-with-actual-time" data-toc-modified-id="Compare-estimation-with-actual-time-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Compare estimation with actual time</a></span><ul class="toc-item"><li><span><a href="#Explore-costs-for-different-edges" data-toc-modified-id="Explore-costs-for-different-edges-1.3.1"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>Explore costs for different edges</a></span></li><li><span><a href="#Determine-dependence-of-estimaiton-vs-time" data-toc-modified-id="Determine-dependence-of-estimaiton-vs-time-1.3.2"><span class="toc-item-num">1.3.2&nbsp;&nbsp;</span>Determine dependence of estimaiton vs time</a></span></li></ul></li><li><span><a href="#Analyze-simulator-FLOP/s" data-toc-modified-id="Analyze-simulator-FLOP/s-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Analyze simulator FLOP/s</a></span><ul class="toc-item"><li><span><a href="#Plot-time-vs-estimated-FLOP" data-toc-modified-id="Plot-time-vs-estimated-FLOP-1.4.1"><span class="toc-item-num">1.4.1&nbsp;&nbsp;</span>Plot time vs estimated FLOP</a></span></li><li><span><a href="#Estimate-FLOP/s" data-toc-modified-id="Estimate-FLOP/s-1.4.2"><span class="toc-item-num">1.4.2&nbsp;&nbsp;</span>Estimate FLOP/s</a></span></li><li><span><a href="#Compare-with-matrix-multiplication" data-toc-modified-id="Compare-with-matrix-multiplication-1.4.3"><span class="toc-item-num">1.4.3&nbsp;&nbsp;</span>Compare with matrix multiplication</a></span></li></ul></li></ul></li><li><span><a href="#CLI" data-toc-modified-id="CLI-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>CLI</a></span></li><li><span><a href="#Nbdev-stuff" data-toc-modified-id="Nbdev-stuff-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Nbdev stuff</a></span></li></ul></div>
# -

# # Compare qtensor with statevector simulations

# +
#export
import sys
import numpy as np
import matplotlib.pyplot as plt

import qtensor as qt
from cartesian_explorer import Explorer
# -

#export
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#db503d', '#02C6E0'])

#export
ex = Explorer()

# ## Qtensor simulation times

# ### Get cost estimation

N = 1000
p = 4
edge_idx = 7
   

# +
 
gamma, beta = [.1]*p, [.3]*p
graph = qt.toolbox.random_graph(nodes=N, degree=3)

comp = qt.QtreeQAOAComposer(graph, gamma=gamma, beta=beta)

comp.energy_expectation_lightcone(list(graph.edges())[edge_idx])
tn = qt.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(comp.circuit)
opt = qt.optimisation.Optimizer.DefaultOptimizer()
peo, _ = opt.optimize(tn)
costs, mems = tn.simulation_cost(peo)
# -

plt.plot(costs, label='flops')
plt.plot(mems, label='memory')
plt.yscale('log')
plt.legend()
plt.grid()
plt.title('Simulation cost per step')
print(f'Total FLOPS={sum(costs)/1e9} G, Memory={max(mems)/1e9} G')

# ### Simulate the circuit

# +
backend = qt.PerfNumpyBackend(print=False)
sim = qt.QtreeSimulator(bucket_backend=backend)

sim.simulate(comp.circuit)
# -

# ### Compare estimation with actual time

# +
profile_results = backend._profile_results
step_times = [x[1] for x in profile_results.values()]
ignored_vars = tn.bra_vars+tn.ket_vars

plt.plot(step_times[len(ignored_vars):], label='Time')
plt.yscale('log')
plt.grid()

plt.twinx()
plt.plot(costs, label='Theoretical FLOPS', color='blue')
plt.yscale('log')

plt.title('Time in simulation for each step')

lastN = 200
axins = plt.gca().inset_axes([0.1, 0.5, 0.47, 0.47])
axins.plot(step_times[-lastN:] )
axins.set_yscale('log')
axins.grid()

plt.twinx(axins)
plt.plot(costs[-lastN:], color='blue')
plt.yscale('log')


print(f'Total time={sum(step_times)}')


# -

# #### Explore costs for different edges

# +
#export
@ex.provider
def graph(n, d, seed):
    return qt.toolbox.random_graph(nodes=n, degree=d, seed=seed)

@ex.provider
def circuit(graph, edge_idx, p, composer_type='cone'):
    gamma, beta = [.1]*p, [.3]*p
    if composer_type=='cylinder':
        comp = qt.OldQtreeQAOAComposer(graph, gamma=gamma, beta=beta)
    if composer_type=='cone':
        comp = qt.QtreeQAOAComposer(graph, gamma=gamma, beta=beta)
    comp.energy_expectation_lightcone(list(graph.edges())[edge_idx])
    return tuple(comp.circuit)

@ex.provider
def tn(circuit):
    return qt.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(circuit)

@ex.provider
def peo(tn):
    opt = qt.optimisation.Optimizer.DefaultOptimizer()
    peo, _ = opt.optimize(tn)
    return tuple(peo)

@ex.provider
def sim_costs(tn, peo):
    opt = qt.optimisation.Optimizer.DefaultOptimizer()
    peo, _ = opt.optimize(tn)
    costs, mems = tn.simulation_cost(peo)
    return costs, mems

@ex.provider
def sum_flops(sim_costs):
    flops, mems = sim_costs
    return sum(flops)


# +
#export
@ex.provider
def step_flops(sim_costs):
    flops, mems = sim_costs
    return tuple(flops)

@ex.provider
def max_mem(sim_costs):
    flops, mems = sim_costs
    return max(mems)


# -

#export
SEED=107

fig = ex.plot_variables2d(('sum_flops', 'max_mem'), n=[N], p=[3],
                          d=[3,4], edge_idx=range(30),
                          seed=[SEED]
                         )
for ax in fig.axes:
    ax.set_yscale('log')
    ax.grid()

# #### Determine dependence of estimaiton vs time

# Determine easy edges from previous graphs.
#

# +
#export
EDGE_IDX_FOR_SEED = {
    107:  [2, 3, 10, 15]
}

EDGE_IDX_FOR_SEED_JLSE = {
    107:  [2, 4, 8, 14, 15, 21]
}
# -

edge_indices = EDGE_IDX_FOR_SEED[SEED]
ds = [3, 4]
p = 3


# +
#export
@ex.provider
def sim_profile(circuit, tn):
    backend = qt.PerfNumpyBackend(print=False)
    sim = qt.QtreeSimulator(bucket_backend=backend)

    sim.simulate(circuit)
    data = backend._profile_results
    return tuple(tuple([tuple(x[0]), x[1]]) for x in data.values())

@ex.provider
def step_sim_time(sim_profile, tn):
    ignored_vars = tn.bra_vars+tn.ket_vars
    times = [x[1] for x in sim_profile]
    return tuple(times[len(ignored_vars):])


# -

f = ex.draw_dependency_graph(figsize=(7,6), node_size=20)

estimators = ex.map_variable('step_flops', d=ds, edge_idx=edge_indices, n=[N], p=[p])

times = ex.map_variable('step_sim_time', d=ds, edge_idx=edge_indices, n=[N], p=[p])

est_flat = np.concatenate(estimators.flatten())
times_flat = np.concatenate(times.flatten())
filt = times_flat<1e-1
plt.scatter(est_flat[filt], times_flat[filt])
plt.grid()
plt.xlabel('estimated FLOP')
plt.ylabel('Runtime')


# ### Analyze simulator FLOP/s

# #### Plot time vs estimated FLOP

#export
def plot_with_filter(est_flat, times_flat):
    filt = (est_flat>1e4) #& (times_flat>1e-4)
    est_flat_filtered = est_flat[filt]
    times_flat_filtered = times_flat[filt]

    # Fit times
    log_fit_coef = np.polyfit(np.log(est_flat_filtered), np.log(times_flat_filtered), 1)
    fit_coef = np.polyfit(est_flat_filtered, times_flat_filtered, 1)
    print('Lin fit:', fit_coef)
    print('Log fit:', log_fit_coef)
    fit_fn = np.poly1d(log_fit_coef)

    # Plot scatter with filtered data
    plt.scatter(est_flat_filtered, times_flat_filtered)
    xfit = 10**np.linspace(4, 7, 100)
    plt.plot(xfit, np.exp(fit_fn(np.log(xfit))), color='blue')
    plt.loglog()
    plt.xlabel('estimated FLOP')
    plt.ylabel('Runtime')
    return log_fit_coef, fit_coef


log_fit_coef, fit_coef = plot_with_filter(est_flat, times_flat)

# Let's estimate our factual FLOPS on a laptop from worst-case on linear plot
#

# #### Estimate FLOP/s

FLOP = 1e6/.03
print(f'Factual FLOPS on a laptop {FLOP:e}')

# Or use logarithmic fit
#

FLOP_logfit = np.exp(-log_fit_coef[1])
print(f'Factual FLOPS on a laptop, from log fit {FLOP_logfit:e}')

N = 500
matmul_flop = N**2*(N-1)
x, y = np.random.randn(2, N, N)
# %timeit np.matmul(x,y)

# #### Compare with matrix multiplication

FLOPS_matmul = matmul_flop/4.65e-3
print(f'FLOPS on this laptop for matrix mul: {FLOPS_matmul:e}')

print(f'Simulator inefficiency: {FLOPS_matmul/FLOP_logfit}')
print(f'Simulator optimality: {FLOP_logfit/FLOPS_matmul}')

#export
import timeit
def get_log_flops_vs_matmul(log_fit_coef):
    FLOPS_logfit = np.exp(-log_fit_coef[1])

    N = 300
    matmul_flop = N**2*(N-1)
    x, y = np.random.randn(2, N, N)
    number = 100
    matmul_time = timeit.timeit(lambda: np.matmul(x,y)
                               , number=number)/number

    FLOPS_matmul = matmul_flop/matmul_time
    
    return FLOPS_logfit, FLOPS_matmul


# ## CLI
#
# Use click to run some scripts from this notebook

# +
# export
from qtensor_specs import cli, click

@cli.command()
@click.argument('filename')
def time_vs_flops_plot(filename):
    """
    Plots times and estimated FLOP for each step of several QAOA energy computation contractions.
    
    Currently using 
        - random regular graphs with degree=3,4 
        - p = 3
        - N = 1000
        
    """
    edge_indices = EDGE_IDX_FOR_SEED[SEED]
    ds = [3, 4]
    p = 3
    N = 1000
    
    estimators = ex.map_variable('step_flops', d=ds,
                                 edge_idx=edge_indices, n=[N], p=[p], seed=[SEED])
    maxmems = ex.map_variable('max_mem', d=ds,
                                 edge_idx=edge_indices, n=[N], p=[p], seed=[SEED])
    if np.max(maxmems)>1e10:
        print('memory estimations:', maxmems)
        raise Exception('Will get too large tetsors!!')
        
    times = ex.map_variable('step_sim_time', d=ds,
                            edge_idx=edge_indices, n=[N], p=[p], seed=[SEED])
    
    est_flat = np.concatenate(estimators.flatten())
    times_flat = np.concatenate(times.flatten())
    
    log_fit_coef, fit_coef = plot_with_filter(est_flat, times_flat)
    plt.savefig(filename)
    
    fit, matmul = get_log_flops_vs_matmul(log_fit_coef)
    
    print('===Results===')
    print(f'Simulator fitted flops: {fit/1e9:.5} G')
    print(f'Matmul flops: {matmul/1e9:.5} G')
    print(f'Simulator optimality: {fit/matmul}')
# -

# ## Nbdev stuff



# +
#hide
#default_exp time_vs_flop

from nbdev.export  import *
notebook2script(fname='Time_vs_FLOP.ipynb')
# -


