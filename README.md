# quadraticOT
Quadratic OT Simulations

Install local package via

pip3 install -3 .

To navigate this repo:

- The "core" directory contains central helpers used in testing
- The "finite_distributions" directory constructs generic objects for manipulating finite, discrete probability distributions
- The "sinkhorn" directory implements the Sinkhorn Phi-regularized algorithm in generality. Also has optimized entropic and quadratic runners. See SinkhornKernels.py for API calls to easily construct Sinkhorn runners.
- The "visualizer" directory contains helper(s) for visualizing joint distributions
- The "examples" directory contains examples of using the Sinkhorn directory. gaussian_dist.ipynb is an easy place to start.
