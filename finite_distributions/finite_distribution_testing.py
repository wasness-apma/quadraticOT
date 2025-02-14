from FiniteDistribution import FiniteDistribution
from core.require import require, requireApproxEq
import numpy as np


def test_init():
    good_distribution_keys = {"a": 0.1, "b": 0.9, "c": 0.0}
    good_distribution = FiniteDistribution(good_distribution_keys)

    bad_distributions = [
        {"a": -0.1, "b": 1.1},
        {"a": 1.1, "b": 0.5},
        {"a": 0.6, "b": 0.7}
    ]
    
    for distribution in bad_distributions:
        didNotFail = False
        try:
            _ = FiniteDistribution(bad_distributions)
            didNotFail = True
        except:
            # do nothing
            pass

        if didNotFail:
            raise Exception(f"Failed on Distribution {distribution}")
        
def test_probability_query():
    distribution_keys = {"a": 0.1, "b": 0.9, "c": 0.0}
    distribution = FiniteDistribution(distribution_keys)

    for key in distribution_keys:
        require(distribution.get_probability(key) == distribution_keys[key])
    
    requireApproxEq(distribution.get_event_probability(lambda key: key == "a" or key == "b"), 1.0)
    requireApproxEq(distribution.get_event_probability(lambda key: key == "a" or key == "c"), 0.1)
    requireApproxEq(distribution.get_event_probability(lambda key: key == "b" or key == "c"), 0.9)
    requireApproxEq(distribution.get_event_probability(lambda key: True), 1.0)
    requireApproxEq(distribution.get_event_probability(lambda key: False), 0.0)

def run_all_tests():
    test_init()
    test_probability_query()
    print("Ran all Tests for FiniteDistributionTesting")

if __name__ == "__main__":
    run_all_tests()

    distribution_keys = {"a": 0.1, "b": 0.3, "c": 0.0, "d": 0.3, "e": 0.2, "f": 0.05, "g": 0.05}
    distribution = FiniteDistribution(distribution_keys)
    distribution.generateBarChart()


    


            


