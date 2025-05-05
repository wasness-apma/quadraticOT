from typing import Callable, Any, Set, Tuple, Dict, override
from finite_distributions.FiniteDistribution import FiniteDistribution
from core.require import require
from sinkhorn.SinkhornRunner import SinkhornRunner
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial


# A runner made for quadratically regularized optimal transport.
# c is the cost function on X \times Y.
class SinkhornRunnerEntropic(SinkhornRunner):
    def __init__(self, cost: Callable[[Any, Any], float]):
        self.cost = cost

            # Phi = lambda x: x * np.log(x)
    # PsiPrime = lambda y: np.exp(y - 1)

        p: float = 2.0
        Phi = lambda x: x * np.log(x)
        PsiPrime = lambda y: np.exp(y - 1)

        super().__init__(cost=cost, Phi=Phi, PsiPrime=PsiPrime)

    @override
    def run_sinkhorn(self, rho1: FiniteDistribution, rho2: FiniteDistribution, epsilon: float, precisionDelta: float, dual_potential_precision_mult: float = 0.01, max_iterations: int = None, printInfo: bool = False) -> Tuple[FiniteDistribution, Callable[[Any], float], Callable[[Any], float], float, float]:
        

        self.rho1Keys = list(rho1.get_keys())
        self.rho2Keys = list(rho2.get_keys())
        rho1Size = len(self.rho1Keys)
        rho2Size = len(self.rho2Keys)
        self.rho1KeyToIndex = [{self.rho1Keys[i]: i for i in range(rho1Size)}]
        self.rho2KeyToIndex = [{self.rho2Keys[i]: i for i in range(rho2Size)}]

        rho1Vector = np.array([rho1.get_probability(self.rho1Keys[i]) for i in range(rho1Size)])
        rho2Vector = np.array([rho2.get_probability(self.rho2Keys[i]) for i in range(rho2Size)])

        CNegativeExpOverEps = np.array([
            [np.exp(-self.cost(self.rho1Keys[i], self.rho2Keys[j]) / epsilon) for j in range(len(self.rho2Keys))] for i in range(len(self.rho1Keys))
        ])

        CNegativeExpOverEpsTranspose = np.transpose(CNegativeExpOverEps)


        fExpOverEpsilon = np.array([1.0] * rho1Size)
        gExpOverEpsilon = np.array([1.0] * rho2Size)

        pimapping = [[0.0] * rho2Size] * rho1Size

        total_search_iterations = 0
        outer_iterations = 0
        while True:
            outer_iterations += 1
            if printInfo:
                print(f"Prior outer iterations: {outer_iterations}. inner iterations: {total_search_iterations}.")

            gExpOverEpsilon, additional_iterations_g = self.calculate_dual_potential_quadratic(rho1Vector, fExpOverEpsilon, self.rho2Keys, CNegativeExpOverEpsTranspose, epsilon, printInfo = printInfo)
            if printInfo:
                print(f"Iterations for g: {additional_iterations_g}")

            
            fExpOverEpsilon, additional_iterations_f = self.calculate_dual_potential_quadratic(rho2Vector, gExpOverEpsilon, self.rho1Keys, CNegativeExpOverEps, epsilon, printInfo = printInfo)
            if printInfo:
                print(f"Iterations for f: {additional_iterations_g}")

            total_search_iterations += additional_iterations_g + additional_iterations_f

            # pimapping = {(key1, key2): self.PsiPrime((f[key1] + g[key2] - self.cost(key1, key2)) / epsilon) * rho1.get_probability(key1) * rho2.get_probability(key2) for key1 in rho1.get_keys() for key2 in rho2.get_keys()}
            pimapping = rho1Vector.reshape(-1, 1) * (fExpOverEpsilon.reshape(-1, 1) * CNegativeExpOverEps * gExpOverEpsilon / np.e) * rho2Vector
            rho1_marginals = np.sum(pimapping, axis = 1)
            rho2_marginals = np.sum(pimapping, axis = 0)

            rho1_error = np.sum(np.abs(rho1_marginals - rho1Vector))
            rho2_error = np.sum(np.abs(rho2_marginals - rho2Vector))

            if printInfo:
                print(f"Error: {rho1_error + rho2_error}")
                if outer_iterations % 1 == 0:
                    print(f"outer iterations: {outer_iterations}. inner iterations: {total_search_iterations}. Error: {rho1_error + rho2_error}")
            if rho1_error + rho2_error < precisionDelta or (max_iterations is not None and outer_iterations >= max_iterations):
                break
            # else:
                # print(f"Rho1 Error: {rho1_error}, Rho2 Error: {rho2_error}. f: {f}, g: {g}, PiMapping: {pimapping}")

        piMapping_remapped = {(self.rho1Keys[i], self.rho2Keys[j]): pimapping[i, j] for i in range(rho1Size) for j in range(rho2Size)}
        f_remappped = {self.rho1Keys[i]: np.log(fExpOverEpsilon[i] * epsilon) for i in range(rho1Size)}
        g_remappped = {self.rho2Keys[j]: np.log(gExpOverEpsilon[j]) * epsilon for j in range(rho2Size)}

        returnDistribution = FiniteDistribution(piMapping_remapped, totalProbabilityErrorAllowance = precisionDelta)

        self.rho1Keys = None
        self.rho2Keys = None
        self.rho1KeyToIndex = None
        self.rho2KeyToIndex = None


        return (returnDistribution, f_remappped, g_remappped, total_search_iterations, outer_iterations)

    @override
    def calculate_dual_potential_quadratic(self, rho: np.array, expPotentialOverEpsilon: np.array, dualKeys: list, expNegativeCostMatrixOverEpsilon: np.array, epsilon: float, printInfo: bool = False) -> Tuple[Dict[Any, float], float]:
        # these are the negatives of f(x) - c(x, y).
        factoredCostMapping = rho * expPotentialOverEpsilon * expNegativeCostMatrixOverEpsilon / np.e
        sums = np.sum(factoredCostMapping, axis = 1)
        innerIterations = factoredCostMapping.size
        outputs = 1/sums

        return outputs, innerIterations

if __name__ == "__main__":
    pass