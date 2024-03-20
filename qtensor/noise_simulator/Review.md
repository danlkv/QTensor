1. Recompute previous ensemble
   This feature makes different runs with different K correlated: if our K=100 was particularly good at estimating the result, our K=500 might be biased. It is better to statistically decouple different runs for more reliable results across the runs.


2. Naming is good
   The long names make it easy to visually browse the code and intuitively understand the purpose of it.

3. `from ... import *`
   This is not a good practice. It makes hard to understand what exactly gets imported from the module. This makes it harder to read the code.

4. NoisySimulator design
   Why do methods for noisy simulation not return the vectors/density matrices, but instead save the result to `.self`?

   The call stack is following: `NoisySimulator.simulate_batch_ensemble()` -> `QtreeSimulator.simulate_batch()` ->
   `NoisySimulator._new_circuit()` -> ...
   This is confusing, since by looking at just NoisySimulator it's hard to understand that `_new_circuit()` is called. This is probably a problem in design of Qtreesimulator.
   However, it makes `NoisySimulator.simulate_batch()` behave appropriatly: to return a stochastic sample of the noisy circuit, which is a good thing.

   TBD: Make `NoisySimulator` to have a reference to `Simulator` and initialization would be `NoisySimulator(noise_model, simulator)`. Then in `NoisySimulator.simulate_ensemble` calls `NoisySimulator.simulate_batch()`, which in turn explicitly applies channels, then calls `self.simulator.simulate_batch(modified_circ, ...)`. Will have to explicitly write the `simulate_batch` method.  

5. Code formatting
   Docstrings in python come after method definition and in triple quotes: """Docstring""". See "Google python styleguide".

6. TODO:
   Noise Channel should contain information on which kraus operator to apply (instead of doing a check by name of channel via `error_name`). In this way, we can allow user to specify custom Kraus Operators.