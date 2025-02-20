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
        require (np.abs(totSum - 1) < errorTerm)

    def get_probability(self, key: Any):
        require(key in self.elementMapping)
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
