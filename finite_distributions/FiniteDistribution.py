from typing import Dict, Any, Callable
from core.require import require
import matplotlib.pyplot as plt
import numpy as np

# this is done generically so that we can encode any sort of key on it
class FiniteDistribution:

    def __init__(self, elementMapping: Dict[Any, float]):
        self.elementMapping = elementMapping
        self.elemList = sorted(list(elementMapping.keys())) # sort by whatever sort makes sense
        self.probabilityList = [elementMapping[key] for key in self.elemList]

        totSum = 0
        for probability in self.probabilityList:
            require(0 <= probability and probability <= 1)
            totSum += probability
        require (totSum <= 1)

    def get_probability(self, key: Any):
        require(key in self.elementMapping)
        return self.elementMapping[key]

    def get_event_probability(self, indicator: Callable[[Any], bool]):
        return np.sum([self.elementMapping[key] for key in self.elementMapping if indicator(key)])

    def generateBarChart(self):
        plt.figure()
        plt.bar(self.elemList, self.probabilityList)
        plt.title("Bar Chart.")
        plt.show()

