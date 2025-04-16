from typing import Callable, Any, Set, Tuple, Dict
from finite_distributions.FiniteDistribution import FiniteDistribution
from core.require import require
import numpy as np


# A generic module for running sinkhorn algorithms and outputting duals on finite spaces. 
# c is the cost function on X \times Y, Phi is the Regularizer, Psi is Phi^*, PsiPrime is the Derivative of Psi
class SinkhornRunner:
    def __init__(self, cost: Callable[[Any, Any], float], Phi: Callable[[float], float], PsiPrime: Callable[[float], float]):
        self.cost = cost
        self.Phi = Phi
        # self.PsiInverse = PsiInverse÷
        self.PsiPrime = PsiPrime

    # rho1 is the X distribution
    # rho2 is the Y distribution
    # epsilon the the regularization parameter
    # 
    def run_sinkhorn(self, rho1: FiniteDistribution, rho2: FiniteDistribution, epsilon: float, precisionDelta: float, dual_potential_precision_mult: float = 0.01, max_iterations: int = None, printInfo: bool = False) -> Tuple[FiniteDistribution, Callable[[Any], float], Callable[[Any], float], float, float]:
        require(epsilon > 0)
        require(precisionDelta > 0)


        f = {key: 0.0 for key in rho1.get_keys()}
        g = {key: 0.0 for key in rho2.get_keys()}

        flippedCostMapping = lambda y, x: self.cost(x, y)

        pimapping = {}
        total_search_iterations = 0
        outer_iterations = 0
        while True:
            outer_iterations += 1
            if printInfo:
                print(f"Prior outer iterations: {outer_iterations}. inner iterations: {total_search_iterations}.")
            g, additional_iterations_g = self.calculate_dual_potential(rho1, f, rho2.get_keys(), self.cost, epsilon, precisionDelta * dual_potential_precision_mult, printInfo = printInfo)
            if printInfo:
                print(f"Iterations for g: {additional_iterations_g}")
            f, additional_iterations_f = self.calculate_dual_potential(rho2, g, rho1.get_keys(), flippedCostMapping, epsilon, precisionDelta * dual_potential_precision_mult, printInfo = printInfo)
            if printInfo:
                print(f"Iterations for f: {additional_iterations_g}")

            total_search_iterations += additional_iterations_g + additional_iterations_f

            pimapping = {(key1, key2): self.PsiPrime((f[key1] + g[key2] - self.cost(key1, key2)) / epsilon) * rho1.get_probability(key1) * rho2.get_probability(key2) for key1 in rho1.get_keys() for key2 in rho2.get_keys()}

            rho1_error = np.sum([np.abs(np.sum([pimapping[(key1, key2)] for key2 in rho2.get_keys()]) - rho1.get_probability(key1)) for key1 in rho1.get_keys()])
            rho2_error = np.sum([np.abs(np.sum([pimapping[(key1, key2)] for key1 in rho1.get_keys()]) - rho2.get_probability(key2)) for key2 in rho2.get_keys()])

            if printInfo:
                print(f"Error: {rho1_error + rho2_error}")
                if outer_iterations % 1 == 0:
                    print(f"outer iterations: {outer_iterations}. inner iterations: {total_search_iterations}. Error: {rho1_error + rho2_error}")
            if rho1_error + rho2_error < precisionDelta or (max_iterations is not None and outer_iterations >= max_iterations):
                break
            # else:
                # print(f"Rho1 Error: {rho1_error}, Rho2 Error: {rho2_error}. f: {f}, g: {g}, PiMapping: {pimapping}")

        returnDistribution = FiniteDistribution(pimapping, totalProbabilityErrorAllowance = precisionDelta)

        return (returnDistribution, f, g, total_search_iterations, outer_iterations)

    def calculate_dual_potential(self, rho: FiniteDistribution, potential: Dict[Any, float], dualKeys: Set[Any], orderedCost: Callable[[Any, Any], float], epsilon: float, dualPrecisionDelta: float, printInfo: bool = False) -> Tuple[Dict[Any, float], float]:
        primalKeys = rho.get_keys()
        dualPotentialMapping: Callable[[Any], float] = {}

        def calculateSumForKey(dualKey: Any, testValue: float):
            return np.sum(
                [self.PsiPrime((potential[primalKey] + testValue - orderedCost(primalKey, dualKey)) / epsilon) * rho.get_probability(primalKey) for primalKey in primalKeys]
            )

        total_iterations = 0
        for dualKey in dualKeys:
            lb = -1.0
            ub = 1.0
            
            while calculateSumForKey(dualKey, lb) > 0.9:
                # print(f"LB {lb} too small")
                lb = lb * 2
                total_iterations += 1
            while calculateSumForKey(dualKey, ub) < 1.1:
                # print(f"UB {ub} too small")
                ub = ub * 2
                total_iterations += 1
            # print(f"lb = {lb}, ub = {ub}")
            
            lbval = calculateSumForKey(dualKey, lb)
            ubval = calculateSumForKey(dualKey, ub)
            while True:
                lamb = 0.5 # (1 - lbval) / (ubval - lbval) if lbval < ubval else 0.5
                x = lb * (1 - lamb) + ub * lamb

                calculated = calculateSumForKey(dualKey, x)
                # print(f"LB: {lb}, UB: {ub}, X: {x}, Calculated: {calculated}")
                # print(f"x = {x}. Calculated = {calculated}.")
                if np.abs(calculated - 1) < dualPrecisionDelta:
                    break
                elif calculated > 1:
                    ub = x 
                    ubval = calculated
                else:
                    lb = x
                    lbval = calculated
                total_iterations += 1
            dualPotentialMapping[dualKey] = x
        return dualPotentialMapping, total_iterations


if __name__ == "__main__":
    pass