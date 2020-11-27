import sys
import time
#print(sys.path)
sys.path.append('.')
import click
import numpy as np
import networkx as nx
from .qaoa import single_qaoa_query

@click.group()
def cli():
    pass

@cli.command()
@click.option('-s','--seed', default=42)
@click.option('-d','--degree', default=3)
@click.option('-n','--nodes', default=10)
@click.option('-p','--p', default=1)
@click.option('-G','--graph-type', default='random_regular')
def query_energy(nodes, p, degree, seed=100, graph_type='random_regular'):
    np.random.seed(seed)
    if graph_type=='random_regular':
        G = nx.random_regular_graph(degree, nodes)
    elif graph_type=='erdos_renyi':
        G = nx.erdos_renyi_graph(nodes, degree/(nodes-1))
    else:
        raise Exception('Unsupported graph type')

    # this is a circular loop graph with max cut
    start = time.time()
    e = single_qaoa_query(G, p)
    end = time.time()
    print('Time elapsed:', end-start)

if __name__=='__main__':
    cli()
