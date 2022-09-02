import numpy as np
import time
from qtensor.optimisation import (
    Optimizer, GreedyOptimizer, RGreedyOptimizer,
    TamakiOptimizer, WithoutOptimizer
)

# Expected time to contract by width
_contraction_time_map = {
    20:0.1,
    21:0.2,
    22:0.4,
    23:1,
    24:2,
    25:4,
    26:8,
    27:20,
    28:40,
    29:80,
    30:150,
    31:300,
    32:600,
    33:1500,
    34:2500,
}

def expected_contraction_time(width):
    """
    Returns expected contraction time in seconds
    """
    keys = list(_contraction_time_map.keys())
    if width < min(keys):
        return 0.
    if width > max(keys):
        return np.inf

    return _contraction_time_map[width]

def should_optimize_more(t_contract, t_opt):
    return t_contract > t_opt*1.5


class AdaptiveOptimizer(Optimizer):

    def log_progress(self, rt,  opt, etime):
        width = opt.treewidth
        opt_name = opt.__class__.__name__
        if hasattr(self, 'verbose'):
            print(f"Qtensor adaptive optimizer: Time={rt:.1f}, width={width}, optimizer={opt_name}, expected contraction time={etime}")

    def optimize(self, tensor_net):
        start = time.time()
        naive = WithoutOptimizer()
        # first, optimize with naive ordering and check treewidth
        res = naive.optimize(tensor_net)

        e1 = expected_contraction_time(naive.treewidth)
        self.log_progress(time.time()-start, naive, e1)

        if not should_optimize_more(e1, time.time()-start):
            self.treewidth = naive.treewidth
            return res


        # Next, greedy
        opt = GreedyOptimizer()
        res = opt.optimize(tensor_net)

        e1 = expected_contraction_time(opt.treewidth)
        self.log_progress(time.time()-start, opt, e1)

        if not should_optimize_more(e1, time.time()-start):
            self.treewidth = opt.treewidth
            return res


        # Next, rgreedy
        rgreedy_time = expected_contraction_time(opt.treewidth-1)
        while rgreedy_time<5:
            opt = RGreedyOptimizer(temp=.02, max_time=rgreedy_time)
            res = opt.optimize(tensor_net)

            e1 = expected_contraction_time(opt.treewidth)
            self.log_progress(time.time()-start, opt, e1)

            if not should_optimize_more(e1, time.time()-start):
                self.treewidth = opt.treewidth
                return res

            rgreedy_time = expected_contraction_time(opt.treewidth-1)

        # Next, Tamaki
        max_simulatable = 32
        width = min(max_simulatable, opt.treewidth-1)
        while True:
            wait_time = expected_contraction_time(width)
            opt = TamakiOptimizer(max_width=width, wait_time=wait_time)
            res = opt.optimize(tensor_net)
            self.treewidth = opt.treewidth
            e1 = expected_contraction_time(opt.treewidth)
            self.log_progress(time.time()-start, opt, e1)

            if not should_optimize_more(e1, time.time() - start):
                return res
            width = opt.treewidth - 1

        return res

