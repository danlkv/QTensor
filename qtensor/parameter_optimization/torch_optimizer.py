#TODO: this file is not used
from . import GradientOptimizer
import torch


class TorchOptimizer(GradientOptimizer):
    def __init__(self, algorithm=None, **kwargs):
        super().__init__(**kwargs)
        self._params = [torch.Tensor(x, requires_grad=True)
                        for x in self._params]

        self._algorithm = algorithm(params=self._params)
        self.loss_history = []
        self.param_history = []

    @property
    def params(self):
        return [x.detach().numpy().data
                for x in self._params]

    def step(self, loss):
        self._algorithm.zero_grad()
        loss.backward()
        self._algorithm.step()


