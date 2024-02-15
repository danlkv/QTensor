#!/usr/bin/env python3
import sys
from pathlib import Path
from functools import wraps
import fire
def log(*args):
    print(f"[main.py] ", *args, file=sys.stderr, flush=True)

# -- Utils

import pandas as pd
import fsspec
import itertools
from dataclasses import dataclass
import io

@dataclass
class File:
    path: Path
    f: io.IOBase

def general_glob(urlpath, **kwargs):
    """General glob function to handle local and remote paths."""
    filelist = fsspec.open_files(urlpath, **kwargs)
    for file in filelist:
        yield file

def is_sequence(x):
    if isinstance(x, str):
        return False
    try:
        iter(x)
        return True
    except TypeError:
        return False

def dict_vector_iter(**d):
    """
    For each value that is a list in dict d, iterate over all possible
    combinations of values.
    """
    keys = d.keys()
    vals = d.values()
    vector_keys = [k for k, v in zip(keys, vals) if is_sequence(v)]
    vector_vals = [v for v in vals if is_sequence(v)]
    for instance in itertools.product(*vector_vals):
        p = dict(d)
        p.update(zip(vector_keys, instance))
        yield p

def general_indexed(in_path, out_path, func, fsspec_kwargs={},  **kwargs):
    """
    Arguments:
        in_path: a glob-like urlpath to pass to fsspec.open_files
        out_path: a string to store the output into. Optionally,
            can provide formatting arguments
            If no formatting arguments provided, will be treated as a directory,
            I.E `<out_path>/{in_file}`
            otherwise, will be treated as a file, I.E. `<out_path>.format(**kwargs)`
            For many input files, the {in_file} argument will be provided.
            This will be passed as the second argument to the function
        func: a function that takes two arguments, the first being the input
            file object, and the second being the output file.
        fsspec_kwargs: kwargs to pass to fsspec.open_files
    """
    # If no formatting arguments provided, treat as directory
    if "{" not in out_path:
        out_pattern = f"{out_path}/{{in_file}}"
    else:
        out_pattern = out_path

    def unit(kwargs):
        in_file = kwargs.pop("in_file")
        in_path = Path(in_file.path)
        out_file = out_pattern.format(
            in_path=in_path,
            in_file=in_path.name,
            **kwargs)
        out_path = Path(out_file)
        # make parent dir
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with in_file.open() as f:
            fl = File(in_path, f)
            changed_out = func(fl, out_file, **kwargs)
        
        log(f"{in_file.path} -> [{func.__name__}] -> {changed_out}")
        index_file = Path(changed_out).parent / "index.csv"
        update_index(index_file, input=in_file.path, output=changed_out, **kwargs)
        return changed_out

    
    in_path = in_path.format(**kwargs)
    files = iter(general_glob(in_path, **fsspec_kwargs))
    combinations = iter(dict_vector_iter(in_file=files, **kwargs))
    return list(map(unit, combinations))

def update_index(index_file, **kwargs):
    df = pd.DataFrame(kwargs, index=[0])
    # check if index file exists
    if not (file := Path(index_file)).exists():
        # create directories if needed
        file.parent.mkdir(parents=True, exist_ok=True)

        print("Creating index file")
        df.to_csv(index_file, header=True, index=False)
    else:
        df_exist = pd.read_csv(index_file, nrows=2)
        if isinstance(df_exist, pd.DataFrame):
            if df_exist.columns.tolist() != df.columns.tolist():
                raise ValueError("Index file already exists but has different columns")
        # append to csv
        print(f"Appending to index file {index_file}")
        df.to_csv(index_file, mode="a", header=False, index=False)
# --

from src.simulators.qtensor import preprocess as qtensor_preprocess
from src.simulators.qtensor import estimate as qtensor_estimate
from src.simulators.qtensor import simulate as qtensor_simulate
from src.simulators.qtensor_energy import simulate as qtensor_simulate_energy
from src.simulators.qtensor_energy import preprocess as qtensor_preprocess_energy
from src.circuit_gen.qaoa import generate_maxcut

# -- Main
sim_preprocessors = {
    'qtensor': qtensor_preprocess,
    'qtensor_energy': qtensor_preprocess_energy
}

sim_estimators = {
    'qtensor': qtensor_estimate
}

sim_simulators = {
    'qtensor': qtensor_simulate,
    'qtensor_energy': qtensor_simulate_energy
}

circ_generators = {
    'qaoa_maxcut': generate_maxcut
}
class Main:

    def echo(self, in_path, out_dir, **kwargs):
        """
         Simple mapper that just echoes stuff
         """
        @wraps(self.echo)
        def unit(in_file, out_file, **kwargs):
            with open(out_file, "wb") as f:
                f.write(in_file.f.read())
            return out_file
        general_indexed(in_path, out_dir, unit, **kwargs)

    def generate(self, out_dir, type, **kwargs):
        @wraps(self.generate)
        def unit(in_file, out_file, type, **kwargs):
            circ_generators[type](out_file, **kwargs)
            return out_file
        general_indexed('/dev/null', out_dir, unit, type=type, **kwargs)

    def preprocess(self, in_path, out_dir, sim='qtensor', **kwargs):
        @wraps(self.preprocess)
        def unit(in_file, out_file, sim, **kwargs):
            sim_preprocessors[sim](in_file, out_file, **kwargs)
            return out_file
        general_indexed(in_path, out_dir, unit, sim=sim, **kwargs)

    def estimate(self, in_path, out_dir, sim='qtensor', **kwargs):
        """
        Estimate the parameters of a simulator
        """
        @wraps(self.estimate)
        def unit(in_file, out_file, sim, **kwargs):
            sim_estimators[sim](in_file, out_file, **kwargs)
            return out_file
        general_indexed(in_path, out_dir, unit, sim=sim, **kwargs)

    if estimate.__doc__:
        # Modify doc to include info about additional parameters
        estimate.__doc__ += f"\n{qtensor_estimate.__doc__.replace('Arguments:', 'Additional:')}"

    def simulate(self, in_path, out_dir, sim='qtensor', **kwargs):
        """
        Simulate the quantum circuit
        """
        @wraps(self.simulate)
        def unit(in_file, out_file, **kwargs):
            sim_simulators[sim](in_file, out_file, **kwargs)
            return out_file
        general_indexed(in_path, out_dir, unit, sim=sim, **kwargs)


if __name__ == "__main__":
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(Main)
