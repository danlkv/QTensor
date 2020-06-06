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
# <div class="toc"><ul class="toc-item"><li><span><a href="#Analyse-chopping" data-toc-modified-id="Analyse-chopping-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Analyse chopping</a></span><ul class="toc-item"><li><span><a href="#Optimal-chops-search-for-task" data-toc-modified-id="Optimal-chops-search-for-task-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Optimal chops search for task</a></span></li><li><span><a href="#Plot-experimental-data" data-toc-modified-id="Plot-experimental-data-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Plot experimental data</a></span></li></ul></li></ul></div>
# -

import sys
sys.path.append('..')
sys.path.append('./qaoa')

# +
import copy
import numpy as np
import networkx as nx
from tqdm import tqdm
from multiprocessing.dummy import Pool

import matplotlib.pyplot as plt
import seaborn as sns

import utils_qaoa as qaoa
import utils
import memcached_qaoa_utils as cached_utils
import pyrofiler as prof
import qtree

sns.set_style('whitegrid')
# %load_ext autoreload
# %autoreload 2
# -


# # Analyse chopping
#

# +
chop_pts = 3
def get_chop_idxs(s, peo, cost, nghs):
    drop_idx = get_chop_dn_drop(nghs)
    min_idx = np.argmin(cost[0][:drop_idx])
    before_min = min_idx - (drop_idx-min_idx)
    on_plato = 2 * min_idx // 3
        
    return min_idx, drop_idx, drop_idx+5

def _cost_before_chop(idxs, cost):
    mems, floats = cost
    before_mem = [max(mems[:i]) for i in idxs]
    return before_mem

def get_chop_dn_drop(nghs):
    nghs = np.array(nghs)
    dn = nghs[1:] - nghs[:-1]
    neg_idx = [i for i, n in enumerate(dn) if n<0]
    pos_idx = [i for i, n in enumerate(dn) if n>0]
    drop_idx = neg_idx[0]
    pos_idx.reverse()
    before_drop = [i for i in pos_idx if i<drop_idx]
    return before_drop[0] - 1


# -

peo, nghs = cached_utils.neigh_peo(23)
costs = cached_utils.graph_contraction_costs(23, peo)
utils.plot_cost(*costs)

# +
sizes = np.arange(25,35)

tasks = [cached_utils.qaoa_expr_graph(s) for s in sizes]
graphs, qbit_sizes = zip(*tasks)

# -

print('Qubit sizes', qbit_sizes)
pool = Pool(processes=20)


peos_n = pool.map(cached_utils.neigh_peo, sizes)
peos, nghs = zip(*peos_n)

with prof.timing('Get full costs naive'):
    costs = pool.starmap(cached_utils.graph_contraction_costs, zip(sizes, peos))

chop_idxs = [
    _idx
    for s, peo, cost, ng in tqdm( zip(sizes, peos, costs, nghs) )
    for _idx in get_chop_idxs(s, peo, cost, ng)
]
chopped_g = [
    cached_utils.contracted_graph(s, peo, _idx)
    for s, peo, cost, ng in tqdm( zip(sizes, peos, costs, nghs) )
    for _idx in get_chop_idxs(s, peo, cost, ng)
]

# +
n_ = 2
G = chopped_g[3*n_+1]
Gc = copy.deepcopy(G)
color_map = []
chopped_peo = peos[n_][chop_idxs[n_*3+1]:]
qtree.graph_model.eliminate_node(Gc, chopped_peo[0])
for node in G:
    if node in chopped_peo[:5]:
        color_map.append('red')
    elif node in chopped_peo:
        color_map.append('blue')
    
plt.figure(figsize=(15,7))
plt.subplot(121)
nx.draw_kamada_kawai(chopped_g[3*n_+1], node_color=color_map, node_size=7, label_size=2)
plt.subplot(122)
nx.draw_kamada_kawai(chopped_g[3*n_+2], node_size=7)

print(sorted(Gc.degree, key=lambda x: x[1]))

# +
print(peos[n_][chop_idxs[n_*3+1]:])
chopped_peo = peos[n_][chop_idxs[n_*3+1]:]

print(sorted(chopped_g[3*n_ + 1].degree, key=lambda x: x[1]))
for n in chopped_peo:
    print(G.degree[n], list(G[n].keys()))
# -

costs_before_chop = [
    mem
    for g, peo, cost, ng in tqdm( zip(graphs, peos, costs, nghs) )
    for mem in _cost_before_chop(get_chop_idxs(g, peo, cost, ng), cost)
]

# +
print('contracted graphs', [g.number_of_nodes() for g in chopped_g])

print('costs before chop', costs_before_chop)

# +
par_vars = [0,1,2,5, 7, 12]

parallelized_g = [
    g
    for graph in chopped_g
    for parvar in par_vars
    for  _, g in [qtree.graph_model.split_graph_by_metric(graph, n_var_parallel=parvar)]
]
# -

print('parallelised graphs', [g.number_of_nodes() for g in parallelized_g])


def n_peo(graph):
    return utils.get_locale_peo(graph, utils.n_neighbors)
_pg_peos = tqdm(list(zip(parallelized_g, peos_par)))
with prof.timing('peos chopped'):
    peos_par_n = pool.map(n_peo, tqdm(parallelized_g))
peos_par, nghs_par = zip(*peos_par_n)


def get_qbb_peo(graph):
    try:
        peo, tw = qtree.graph_model.get_peo(graph)
        fail = False
    except:
        print('QBB fail, nodes count:', graph.number_of_nodes())
        peo, nghs = utils.get_locale_peo(graph, utils.n_neighbors)
        fail = True
    return peo, fail


peos_par = [ get_qbb_peo(g) for g in tqdm( parallelized_g ) ]
peos_par, fails_qbb = zip(*peos_par)

tqdm._instances.clear()


_pg_peos = tqdm(list(zip(parallelized_g, peos_par)))
with prof.timing('Costs chopped'):
    costs_all = pool.map(_get_cost, _pg_peos)


experiment_name = 'small_chops_test'

mems = [max(m) for m,_ in costs_all ]

# +
_data = np.array(mems).reshape(len(sizes), chop_pts, len(par_vars)) 

print(_data)
np.save(f'cached_data/{experiment_name}',_data)


# -

def trid_plot(x, y, labels, dimspec=(0,1,2), figsize=(15,4)): 
    y = y.transpose(dimspec)
    plot_cnt = y.shape[0]
    line_cnt = y.shape[1]
    def _label_of(dim, idx):
        return labels[dim] + ' ' + str(x[dim][idx])
    
    fig, axs = plt.subplots(1, plot_cnt, sharey=True, figsize=figsize)
    try:
        iter(axs)
    except TypeError:
        axs = [axs]
    for i, ax in enumerate(axs):
        plt.sca(ax)
        plt.title(_label_of(0, i))
        for j in range(line_cnt):
            color=plt.cm.jet(j/line_cnt)
            plt.plot(x[2], y[i,j],color=color, label=_label_of(1, j))
            plt.xlabel(labels[2])
            plt.yscale('log')
            plt.minorticks_on()


# %pwd

sizes = np.linspace(55, 65, 40).round(1)
par_vars = [0,1,2,3,5,7,8,9,10,11,12]
chop_pts = 12
experiment_name='skylake_qbb_randomd3s44_55-65x4'
costs = np.load('./cached_data/skylake_qbb_randomreg*d3*s44_55-65x4_costs.npy', allow_pickle=True)
mems, flops= zip(*costs)
memsmax = [max(m) for m in mems]
flopsum = [sum(x) for x in flops]
_data = np.array(memsmax).reshape(len(sizes), chop_pts, len(par_vars))
#_data = np.array(flopsum).reshape(len(sizes), chop_pts, len(par_vars))
print(_data.shape)
_data[0,0]




# +
xs = [np.arange(chop_pts), sizes, par_vars]
names = ['Chop point', 'Task size', 'par vars']

trid_plot(xs, _data, names ,(1,0,2))
plt.suptitle('Parallelisation with chopping, naive peo')
plt.savefig(f'figures/chop_analysis__{experiment_name}.pdf')

# +
xs = [sizes, par_vars, np.arange(chop_pts)]
names = ['Task size', 'Par vars', 'chop part']

trid_plot(xs, _data, names ,(0,2,1))
plt.suptitle('Costs for different chopping pts, naive peo')
plt.legend()
plt.savefig(f'figures/chop_point_analysis__{experiment_name}.pdf')

# +
_chopcost = np.array(costs_before_chop).reshape(len(sizes), chop_pts, 1)
trid_plot([' ', sizes, range(chop_pts)], _chopcost, ['Chop cost', 'Task size', 'Chop part'], (2,0,1))

print(_chopcost)
# -


# ## Optimal chops search for task

optimal = _data.min(axis=1)
optimal = optimal[..., np.newaxis]
print(optimal.shape)

optimal
xs = [' ', sizes[5:25], par_vars]
trid_plot(xs, optimal[5:25], ['Cost scaling, memory max, random regular degree 3', 'Task size', 'Par vars'], 
          (2,0,1), figsize=(6,10))
plt.savefig(f'figures/skylake_optimal_{experiment_name}_mems.pdf')
plt.legend()

optimal
xs = [' ', sizes, par_vars]
trid_plot(xs, optimal, ['Cost scaling, flop sum, random regular degree=3', 'Task size', 'Par vars'],
          (2,0,1), figsize=(6,10))
plt.legend()
plt.savefig(f'figures/skylake_optimal_{experiment_name}_flops.pdf')

# ## Plot experimental data

# +
data = [
    [ 160, 117, 68, 35, 33]
    ,[ 201, 159, 100, 69, 63 ]
    ,[ 267, 247, 187, 165, 189]
]

nodes = [ 2**6, 2**7, 2**8, 2**9, 2**10]
colors = (plt.cm.gnuplot2(x) for x in np.linspace(.2,.8,len(data)))

plt.plot(nodes, data[0], 'D-', label='parallel part time', color=next(colors))
plt.plot(nodes, data[1], 'D--', label='simulation time', color=next(colors))
#plt.plot(nodes, data[2], 'D--', label='total job time', color=next(colors))
plt.xlabel('Nodes count')
plt.ylabel('Time of simulation, seconds')
plt.loglog(basex=2, basey=2)
plt.minorticks_on()
plt.grid(which='minor', alpha=0.3, linestyle='-', axis='both')
plt.legend()
plt.savefig('figures/experimental_node_scaling_58d3.pdf')
# -

_d = np.array(data)
plt.plot(_d[2]-_d[0])

colors = (plt.cm.gnuplot2(x) for x in np.linspace(.2,.8,len(data)))
c = next(colors)

next(colors)

255*.24

