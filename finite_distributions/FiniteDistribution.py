from typing import Dict, Any

class FiniteDistribution:

    def __main__(self, elementMapping: Dict[Any, float]):
        self.elementMapping = elementMapping
        self.elemList = sorted(list(elementMapping.keys())) # sort by whatever sort makes sense
        self.probabilityList = [elementMapping[key] for key in self.elemList]

        totSum = 0
        for probability in self.probabilityList:
            if probability < 0 or elementMapping[elem]


    def generateBarChart

