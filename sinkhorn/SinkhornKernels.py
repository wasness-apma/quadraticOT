from sinkhorn.SinkhornRunner import SinkhornRunner
from sinkhorn.SinkhornRunnerQuadratic import SinkhornRunnerQuadratic
from sinkhorn.SinkhornRunnerEntropic import SinkhornRunnerEntropic
from typing import Callable, Any
import numpy as np
from core.require import require


# call bespoke entropic runner
def get_entropically_regularized_runner(cost: Callable[[Any, Any], float]) -> SinkhornRunner:
    return SinkhornRunnerEntropic(cost)

# Use general runner for $p$-norm
def get_pnorm_regularized_runner(p: float, cost: Callable[[Any, Any], float]) -> SinkhornRunner:
    require(p > 1 and p < np.inf)

    Phi = lambda x: (0 if x <= 0 else (1/p) * np.power(x, p))
    PsiPrime = lambda y: (0 if y <= 0 else np.power(y, 1 / (p - 1)))

    return SinkhornRunner(cost, Phi, PsiPrime)

# call bespoke quadratically regularized runner
def get_quadratically_regularized_runner(cost: Callable[[Any, Any], float], use_parallelization: bool) -> SinkhornRunner:
    return SinkhornRunnerQuadratic(cost, use_parallelization = use_parallelization)