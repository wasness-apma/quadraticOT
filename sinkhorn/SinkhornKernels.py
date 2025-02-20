from sinkhorn.SinkhornRunner import SinkhornRunner
from typing import Callable, Any
import numpy as np
from core.require import require

def get_entropically_regularized_runner(cost: Callable[[Any, Any], float]) -> SinkhornRunner:
    Phi = lambda x: x * np.log(x)
    PsiPrime = lambda y: np.exp(y - 1)

    return SinkhornRunner(cost, Phi, PsiPrime)

def get_pnorm_regularized_runner(p: float, cost: Callable[[Any, Any], float]) -> SinkhornRunner:
    require(p > 1 and p < np.inf)

    Phi = lambda x: (0 if x <= 0 else (1/p) * np.power(x, p))
    PsiPrime = lambda y: (0 if y <= 0 else np.power(y, 1 / (p - 1)))

    return SinkhornRunner(cost, Phi, PsiPrime)