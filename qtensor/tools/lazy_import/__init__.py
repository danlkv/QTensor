import importlib
import sys


class LasyModule:
    def __init__(self, modulename):
        self.modulename = modulename
        self.module = None

    def __getattr__(self, attr):
        if self.module is None:
            try:
                #print(f"LazyModule: importing {self.modulename}", file=sys.stderr)
                self.module = importlib.import_module(self.modulename)
            except (ImportError, ModuleNotFoundError):
                #print(f"LazyModule: {self.modulename} is missing.", file=sys.stderr)
                raise

        return self.module.__getattribute__(attr)



tcontract = LasyModule('tcontract')
torch = LasyModule('torch')
qiskit = LasyModule('qiskit')
qiskit_lib = LasyModule('qiskit.extensions.standard')
mpi4py = LasyModule('mpi4py')
MPI = LasyModule('mpi4py.MPI')
networkit = LasyModule('networkit')
""" can this break something? need to think more

What if user imports qtensor, and then networkit?


if sys.modules.get('networkit') is None:
    sys.modules['networkit'] = networkit
"""
