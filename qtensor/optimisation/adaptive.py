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

def should_optimize_more(t_contract, t_opt, opt_scale=1.5):
    return t_contract > t_opt*opt_scale


class AdaptiveOptimizer(Optimizer):

    def __init__(self, max_time=np.inf, opt_sim_ratio=1.5):
        """
        Optimizes contraction cost with consideration of simulation time.
        The goal is that time on optimization is some multiple (opt_sim_ratio)
        of time for simulation

        Args:
            max_time:
                maximum time to spend in optimization. If finite, this may
                result in breaking the assumption of time tradeof
            opt_sim_ratio:
                ratio of optimization time to simulation time.
        """
        self.max_time = max_time
        self.opt_sim_ratio = opt_sim_ratio

    def log_progress(self, rt, opt, etime, width):
        opt_name = opt.__class__.__name__
        if hasattr(self, 'verbose'):
            print(f"Qtensor adaptive optimizer: Time={rt:.4f}, width={width}, optimizer={opt_name}, expected contraction time={etime}")

    def get_ordering_ints(self, graph, inplace=False):
        start = time.time()
        naive = WithoutOptimizer()
        # first, optimize with naive ordering and check treewidth
        peo, path = naive.get_ordering_ints(graph)
        width = max(path)

        e1 = expected_contraction_time(width)
        self.log_progress(time.time()-start, naive, e1, width)

        if not should_optimize_more(e1, time.time()-start, self.opt_sim_ratio):
            return peo, path


        # Next, greedy
        opt = GreedyOptimizer()
        peo, path = opt.get_ordering_ints(graph)
        width = max(path)

        e1 = expected_contraction_time(width)
        self.log_progress(time.time()-start, opt, e1, width)

        if not should_optimize_more(e1, time.time()-start, self.opt_sim_ratio):
            return peo, path


        # Next, rgreedy
        rgreedy_time = expected_contraction_time(width-1)
        while rgreedy_time<5:
            opt = RGreedyOptimizer(temp=.02, max_time=rgreedy_time)
            peo, path = opt.get_ordering_ints(graph)
            width = max(path)

            e1 = expected_contraction_time(width)
            self.log_progress(time.time()-start, opt, e1, width)

            if not should_optimize_more(e1, time.time()-start, self.opt_sim_ratio):
                return peo, path

            rgreedy_time = expected_contraction_time(width-1)

        # Next, Tamaki
        max_simulatable = 32
        target_width = min(max_simulatable, width-1)
        while True:
            # terminate if reached max time - 1. No sense in running tamaki for 1 second
            # at this scale.
            spent_so_far = time.time() - start
            if spent_so_far > self.max_time:
                print("Adaptive ordering algo exceeded budget of",
                      f"{self.max_time} seconds. Returning prematurely")
                return peo, path
            wait_time = min(
                expected_contraction_time(width),
                # reserve a second for tamaki overhead
                self.max_time - spent_so_far
            )
            # Tamaki may fail to process very large graphs if the budget is too small
            wait_time += 1

            opt = TamakiOptimizer(max_width=target_width, wait_time=wait_time)
            # Detect termination reason. 
            # If terminated because reached max_width, then reduce the width
            # Othervise need more time
            start_opt = time.time()
            t_peo, t_path = opt.get_ordering_ints(graph)
            t_width = max(t_path)
            opt_duration = time.time() - start_opt
            # Record result if it's better than what we already have
            # (Sometimes it can decrease if we are close to time budget)
            if t_width <= target_width:
                peo = t_peo
                path = t_path
                width = t_width

            e1 = expected_contraction_time(width)
            self.log_progress(time.time()-start, opt, e1, width)

            if not should_optimize_more(e1, time.time() - start, self.opt_sim_ratio):
                return peo, path

            # Do not reduce target treewidth if failed to converge to the previous one.
            if opt_duration < wait_time - 1:
                target_width = width - 1


        return peo, path

