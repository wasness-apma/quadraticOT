from typing import Callable, Any
from finite_distributions import FiniteDistribution
from core.require import require


# A generic module for running sinkhorn algorithms and outputting duals on finite spaces. 
# c is the cost function on X \times Y, Phi is the Regularizer, Psi is Phi^*, PsiPrime is the Derivative of Psi
class SinkhornRunner:



    def __init__(self, c: Callable[[float, float], float], Phi: Callable[[float], float], Psi: Callable[[float], float], PsiPrime: Callable[[float], float]):
        self.c = c
        self.Phi = Phi
        self.Psi = Psi
        self.PsiPrime = PsiPrime

    # rho1 is the X distribution
    # rho2 is the Y distribution
    # epsilon the the regularization parameter
    # 
    def run_sinkhorn(rho1: FiniteDistribution, rho2: FiniteDistribution, epsilon: float):
        f_initial = 

    def calculate_dual_potential(rho: FiniteDistribution, potential: Callable[[Any], float], cost: Callable[[Any], float], epsilon: float):
        pass

    
        





if __name__ == "__main__":
    pass