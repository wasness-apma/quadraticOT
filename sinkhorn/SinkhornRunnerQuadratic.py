from typing import Callable, Any, Set, Tuple, Dict, override
from finite_distributions.FiniteDistribution import FiniteDistribution
from core.require import require
from sinkhorn.SinkhornRunner import SinkhornRunner
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial


# A generic module for running sinkhorn algorithms and outputting duals on finite spaces. 
# c is the cost function on X \times Y, Phi is the Regularizer, Psi is Phi^*, PsiPrime is the Derivative of Psi
class SinkhornRunnerQuadratic(SinkhornRunner):
    def __init__(self, cost: Callable[[Any, Any], float], use_parallelization: bool):
        self.cost = cost

        p: float = 2.0
        Phi = lambda x: (0 if x <= 0 else (1/2) * np.power(x, 2.0))
        PsiPrime = lambda y: np.max((y, 0))

        self.use_parallelization = use_parallelization

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

        self.C = np.array([
            [self.cost(self.rho1Keys[i], self.rho2Keys[j]) for j in range(len(self.rho2Keys))] for i in range(len(self.rho1Keys))
        ])

        self.CTranspose = np.transpose(self.C)

        f = np.array([0.0] * rho1Size)
        g = np.array([0.0] * rho2Size)

        pimapping = [[0.0] * rho2Size] * rho1Size

        total_search_iterations = 0
        outer_iterations = 0
        while True:
            outer_iterations += 1
            if printInfo:
                print(f"Prior outer iterations: {outer_iterations}. inner iterations: {total_search_iterations}.")

            
            g, additional_iterations_g = self.calculate_dual_potential_quadratic(rho1Vector, f, self.rho2Keys, self.CTranspose, epsilon, printInfo = printInfo)
            if printInfo:
                print(f"Iterations for g: {additional_iterations_g}")

            
            f, additional_iterations_f = self.calculate_dual_potential_quadratic(rho2Vector, g, self.rho1Keys, self.C, epsilon, printInfo = printInfo)
            if printInfo:
                print(f"Iterations for f: {additional_iterations_g}")

            total_search_iterations += additional_iterations_g + additional_iterations_f

            # pimapping = {(key1, key2): self.PsiPrime((f[key1] + g[key2] - self.cost(key1, key2)) / epsilon) * rho1.get_probability(key1) * rho2.get_probability(key2) for key1 in rho1.get_keys() for key2 in rho2.get_keys()}
            pimapping = rho1Vector.reshape(-1, 1) * np.maximum(f.reshape(-1, 1) - self.C + g, 0) * rho2Vector / epsilon
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
        f_remappped = {self.rho1Keys[i]: f[i] for i in range(rho1Size)}
        g_remappped = {self.rho2Keys[j]: g[j] for j in range(rho1Size)}

        returnDistribution = FiniteDistribution(piMapping_remapped, totalProbabilityErrorAllowance = precisionDelta)

        self.rho1Keys = None
        self.rho2Keys = None
        self.rho1KeyToIndex = None
        self.rho2KeyToIndex = None


        return (returnDistribution, f_remappped, g_remappped, total_search_iterations, outer_iterations)

    @staticmethod
    def calculateDesiredPrimal(costArray: np.array, probabilityArray: np.array, epsilon: float):
        # first sort the Array
        arraySorting = costArray.argsort()
        sorted_arr = costArray[arraySorting]
        sorted_probabilities = probabilityArray[arraySorting]

        sumSoFar = 0.0 # sum so far
        derivativeSoFar = 0.0 # derivative of the previous segment
        newDerivative = 0.0 # initialize in case of triviality

        i = 0
        while i < len(sorted_arr) - 1:
            curX = sorted_arr[i]
            nextX = sorted_arr[i+1]

            newDerivative = derivativeSoFar + sorted_probabilities[i]  # probability associated with primal key
            nextSum = sumSoFar + newDerivative * (nextX - curX)

            if nextSum > epsilon:
                break

            sumSoFar = nextSum
            derivativeSoFar = newDerivative
            i += 1

        if i == len(sorted_arr) - 1:
            #  need to update in case of edge conditions
            curX = sorted_arr[i]
            newDerivative += sorted_probabilities[i]

        x = (epsilon - sumSoFar) / newDerivative + curX
        return x, i

    @override
    def calculate_dual_potential_quadratic(self, rho: np.array, potential: np.array, dualKeys: list, costMatrix: np.array, epsilon: float, printInfo: bool = False) -> Tuple[Dict[Any, float], float]:
        # these are the negatives of f(x) - c(x, y).
        negativePotentialMinusCostMappings = -(potential - costMatrix)
        
        innerIterations = 0

        if not self.use_parallelization:
            outputs = []
            for i in range(len(dualKeys)):
                dualKey = dualKeys[i]
                newValue, newIters = self.calculateDesiredPrimal(negativePotentialMinusCostMappings[i], rho, epsilon)
                outputs.append(newValue)
                innerIterations += newIters
        else:
            partialFuncPrimal = partial(self.calculateDesiredPrimal, probabilityArray=rho, epsilon=epsilon)
            with ProcessPoolExecutor() as executor:
                batches = [negativePotentialMinusCostMappings[i] for i in range(len(dualKeys))]
                results = list(executor.map(partialFuncPrimal, batches))
                outputs = np.array([results[i][0] for i in range(len(batches))])
                innerIterations += sum([res[1] for res in batches])

        return np.array(outputs), innerIterations
            





        

if __name__ == "__main__":
    pass