# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/index.ipynb (unless otherwise specified).

__all__ = ['cli']

# Cell
import click

@click.group()
def cli():
    pass

from qtensor_specs import speed_comparison
from qtensor_specs import qaoa_bench
from qtensor_specs import time_vs_flop
