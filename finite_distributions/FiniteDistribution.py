from typing import Dict, Any, Callable, Set, Optional
from core.require import require
import matplotlib.pyplot as plt
import numpy as np

# this is done generically so that we can encode any sort of key on it
class FiniteDistribution:

    defaultErrorEpsilon: float = 0.000001

    def __init__(self, elementMapping: Dict[Any, float], totalProbabilityErrorAllowance: Optional[float] = None):
        self.elementMapping = elementMapping
        self.elemList = sorted(list(elementMapping.keys())) # sort by whatever sort makes sense
        self.probabilityList = [elementMapping[key] for key in self.elemList]

        errorTerm = totalProbabilityErrorAllowance if totalProbabilityErrorAllowance is not None else self.defaultErrorEpsilon
        totSum = 0
        for probability in self.probabilityList:
            require(-errorTerm <= probability and probability <= 1 + errorTerm)
            totSum += probability
        require (np.abs(totSum - 1) < errorTerm, throwMessage = f"Faulty totSum={totSum} for error tolerance errorTerm={errorTerm}")

    def get_probability(self, key: Any):
        require(key in self.elementMapping, f"Found invalid key {key}. Valid keys: f{list(self.elementMapping.keys())}")
        return self.elementMapping[key]

    def get_event_probability(self, indicator: Callable[[Any], bool]):
        return np.sum([self.elementMapping[key] for key in self.elementMapping if indicator(key)])

    def get_keys(self) -> Set[Any]:
        return set(self.elemList)

    def generateBarChart(self):
        plt.figure()
        plt.bar(self.elemList, self.probabilityList)
        plt.title("Bar Chart.")
        plt.show()

    # Any == FiniteDistribution. Self-Typing is awkward in python.
    def productDistribution(self, other: Any, totalProbabilityErrorAllowance: Optional[float] = None) -> Any:
        elementMapping = {(key1, key2): self.get_probability(key1) * other.get_probability(key2) for key1 in self.elementMapping.keys() for key2 in other.elementMapping.keys()}
        return FiniteDistribution(elementMapping, totalProbabilityErrorAllowance = totalProbabilityErrorAllowance)

    def integrate(self, functional: Callable[[Any], float]) -> float:
        sum = 0
        for element in self.elementMapping:
            sum += functional(element) * self.get_probability(element)
        return sum
    
    # Any == FiniteDistribution. Self-Typing is awkward in python.
    def flip_product_distribution_order(self, totalProbabilityErrorAllowance = None) -> Any:
        new_element_mapping = {(y, x): self.elementMapping[(x, y)] for (x, y) in self.elementMapping}
        return FiniteDistribution(new_element_mapping, totalProbabilityErrorAllowance = totalProbabilityErrorAllowance)

    # if product distribution, calculate the conditional map, assuming RHS is a float
    def calculate_conditional_map(self) -> Dict[Any, float]:
        xs = list(set([x for (x, y) in self.elementMapping]))
        ys = list(set([y for (x, y) in self.elementMapping]))
        conditionalMap = {
            x : np.sum([y * self.elementMapping.get((x, y), 0.0)for y in ys]) / np.sum([self.elementMapping.get((x, y), 0.0)for y in ys]) for x in xs
        }
        return conditionalMap   