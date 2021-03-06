import importlib
import sys


class LasyModule:
    def __init__(self, modulename):
        self.modulename = modulename
        self.module = None

    def __getattr__(self, attr):
        if self.module is None:
            try:
                self.module = importlib.import_module(self.modulename)
            except (ImportError, ModuleNotFoundError):
                print(f"LazyModule: {self.modulename} is missing.", file=sys.stderr)
                raise

        return self.module.__getattribute__(attr)

class FallbackLasyModule:
    def __init__(self, modulenames):
        self.modulenames = modulenames
        self.module = None

    def __getattr__(self, attr):
        if self.module is None:
            err = None
            for modulename in self.modulenames:
                try:
                    self.module = importlib.import_module(modulename)
                except (ImportError, ModuleNotFoundError) as e:
                    err = e

            if self.module is None:
                print(f"LazyModule: {self.modulename} is missing.", file=sys.stderr)
                raise err

        return self.module.__getattribute__(attr)


tcontract = LasyModule('tcontract')
qiskit = LasyModule('qiskit')
qiskit_lib = FallbackLasyModule(['qiskit.circuit.library','qiskit.extensions.standard'])
mpi4py = LasyModule('mpi4py')
MPI = LasyModule('mpi4py.MPI')
