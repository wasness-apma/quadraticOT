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
    def run_sinkhorn(self, rho1: FiniteDistribution, rho2: FiniteDistribution, epsilon: float, precisionDelta: float) -> Tuple[FiniteDistribution, Callable[[Any], float], Callable[[Any], float]]:
        require(epsilon > 0)
        require(precisionDelta > 0)


        f = {key: 0.0 for key in rho1.get_keys()}
        g = {key: 0.0 for key in rho2.get_keys()}

        flippedCostMapping = lambda y, x: self.cost(x, y)

        pimapping = {}
        while True:
            g = self.calculate_dual_potential(rho1, f, rho2.get_keys(), self.cost, epsilon, precisionDelta * 0.01)
            f = self.calculate_dual_potential(rho2, g, rho1.get_keys(), flippedCostMapping, epsilon, precisionDelta * 0.01)

            pimapping = {(key1, key2): self.PsiPrime((f[key1] + g[key2] - self.cost(key1, key2)) / epsilon) * rho1.get_probability(key1) * rho2.get_probability(key2) for key1 in rho1.get_keys() for key2 in rho2.get_keys()}

            rho1_error = np.sum([np.abs(np.sum([pimapping[(key1, key2)] for key2 in rho2.get_keys()]) - rho1.get_probability(key1)) for key1 in rho1.get_keys()])
            rho2_error = np.sum([np.abs(np.sum([pimapping[(key1, key2)] for key1 in rho1.get_keys()]) - rho2.get_probability(key2)) for key2 in rho2.get_keys()])

            if rho1_error + rho2_error < precisionDelta:
                break
            # else:
                # print(f"Rho1 Error: {rho1_error}, Rho2 Error: {rho2_error}. f: {f}, g: {g}, PiMapping: {pimapping}")

        returnDistribution = FiniteDistribution(pimapping, totalProbabilityErrorAllowance = precisionDelta)

        return (returnDistribution, f, g)

    def calculate_dual_potential(self, rho: FiniteDistribution, potential: Dict[Any, float], dualKeys: Set[Any], orderedCost: Callable[[Any, Any], float], epsilon: float, dualPrecisionDelta: float) -> Dict[Any, float]:
        primalKeys = rho.get_keys()
        dualPotentialMapping: Callable[[Any], float] = {}

        def calculateSumForKey(dualKey: Any, testValue: float):
            return np.sum(
                [self.PsiPrime((potential[primalKey] + testValue - orderedCost(primalKey, dualKey)) / epsilon) * rho.get_probability(primalKey) for primalKey in primalKeys]
            )

        for dualKey in dualKeys:
            lb = -1.0
            ub = 1.0
            
            while calculateSumForKey(dualKey, lb) > 0.9:
                lb = lb * 2
            while calculateSumForKey(dualKey, ub) < 1.1:
                ub = ub * 2
            
            while True:
                x = (lb + ub) / 2
                calculated = calculateSumForKey(dualKey, x)
                # print(f"LB: {lb}, UB: {ub}, X: {x}, Calculated: {calculated}")
                if np.abs(calculated - 1) < dualPrecisionDelta:
                    break
                elif calculated > 1:
                    ub = x 
                else:
                    lb = x
            dualPotentialMapping[dualKey] = x
        return dualPotentialMapping
                

            

            


    
        





if __name__ == "__main__":
    pass