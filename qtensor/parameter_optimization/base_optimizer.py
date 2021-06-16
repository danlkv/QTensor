#TODO: this file is not used
class Optimizer:
    def __init__(self, params=[], backend=None):
        self._backend = backend
        self._params = params

    def optimize(self):
            raise NotImplementedError


class GradientOptimizer(Optimizer):
    def __init__(self, algorithm=None, **kwargs):
        super().__init__(**kwargs)
        self._algorithm = algorithm
        self.loss_history = []
        self.param_history = []

    def _get_loss(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    @property
    def params(self):
        return self._params

    def optimize(self, steps=20):

        for i in range(steps):
            loss = self._get_loss()
            self.step(loss)
            self.loss_history.append(loss)
            self.param_history.append(self.params)

