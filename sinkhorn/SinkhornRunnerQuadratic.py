from typing import Callable, Any, Set, Tuple, Dict, override
from finite_distributions.FiniteDistribution import FiniteDistribution
from core.require import require
from sinkhorn.SinkhornRunner import SinkhornRunner
import numpy as np


# A generic module for running sinkhorn algorithms and outputting duals on finite spaces. 
# c is the cost function on X \times Y, Phi is the Regularizer, Psi is Phi^*, PsiPrime is the Derivative of Psi
class SinkhornRunnerQuadratic(SinkhornRunner):
    def __init__(self, cost: Callable[[Any, Any], float]):
        self.cost = cost

        p: float = 2.0
        Phi = lambda x: (0 if x <= 0 else (1/2) * np.power(x, 2.0))
        PsiPrime = lambda y: np.max((y, 0))

        super().__init__(cost=cost, Phi=Phi, PsiPrime=PsiPrime)

    @override
    def calculate_dual_potential(self, rho: FiniteDistribution, potential: Dict[Any, float], dualKeys: Set[Any], orderedCost: Callable[[Any, Any], float], epsilon: float, dualPrecisionDelta: float) -> Tuple[Dict[Any, float], float]:
        primalKeys = rho.get_keys()
        
        # these are the negatives of f(x) - c(x, y).
        negativePotentialMinusCostMappings = {
            dualKey: np.array([[-(potential[primalKey] - orderedCost(primalKey, dualKey)) , rho.get_probability(primalKey)] for primalKey in primalKeys]) for dualKey in dualKeys
        }

        def calculateDesiredPrimal(costArray: list):
            # first sort the Array
            sorted_arr = costArray[costArray[:, 0].argsort()]
            sumSoFar = 0.0 # sum so far
            derivativeSoFar = 0.0 # derivative of the previous segment
            newDerivative = 0.0 # initialize in case of triviality

            i = 0
            while i < len(sorted_arr) - 1:
                curX = sorted_arr[i][0]
                nextX = sorted_arr[i+1][0]

                newDerivative = derivativeSoFar + sorted_arr[i][1]  # probability associated with primal key
                nextSum = sumSoFar + newDerivative * (nextX - curX)

                if nextSum > epsilon:
                    break

                sumSoFar = nextSum
                derivativeSoFar = newDerivative
                i += 1

            if i == len(sorted_arr) - 1:
                #  need to update in case of edge conditions
                curX = sorted_arr[i][0] 
                newDerivative += sorted_arr[i][1]

            x = (epsilon - sumSoFar) / newDerivative + curX
            return x, i
        
        dualPotentialMapping: Callable[[Any], float] = {}
        innerIterations = 0
        for dualKey in dualKeys:
            newValue, newIters = calculateDesiredPrimal(negativePotentialMinusCostMappings[dualKey])
            dualPotentialMapping[dualKey] = newValue
            innerIterations += newIters

        return dualPotentialMapping, innerIterations
            





        

if __name__ == "__main__":
    pass