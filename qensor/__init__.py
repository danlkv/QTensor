from .CircuitComposer import QAOAComposer
from .OpFactory import CirqCreator, QtreeCreator

class CirqQAOAComposer(QAOAComposer, CirqCreator):
    pass

class QtreeQAOAComposer(QAOAComposer, QtreeCreator):
    pass
