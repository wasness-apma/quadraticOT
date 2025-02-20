from finite_distributions.FiniteDistribution import FiniteDistribution
from core.require import require, requireApproxEq
import numpy as np
import sinkhorn.SinkhornKernels as skern


def test_singleton():
    dist_keys = {"a": 1.0}
    dist_a = FiniteDistribution(dist_keys)
    dist_b = FiniteDistribution(dist_keys)

    cost = lambda x, y: 0

    entropic = skern.get_entropically_regularized_runner(cost)
    pnorm = skern.get_pnorm_regularized_runner(7.3, cost)

    for runner in [entropic, pnorm]:
        epsilon = 0.1
        delta = 0.01

        ran = runner.run_sinkhorn(dist_a, dist_b, epsilon, delta)
        require(len(ran[0].elementMapping.keys()) == 1)
        require(list(ran[0].elementMapping.keys())[0] == ("a", "a"))
        require(np.abs(ran[0].get_probability(("a", "a")) - 1) < delta)
        
def test_coin_flip_2norm():
    # test comes from Nutz paper, "Quadratically Regularized Optima Transport: Exitence and Multiplicity of Potentials"
    dist_a_keys = {"H1": 0.5, "T1": 0.5}
    dist_a = FiniteDistribution(dist_a_keys)

    dist_b_keys = {"H2": 0.5, "T2": 0.5}
    dist_b = FiniteDistribution(dist_b_keys)

    # test cost for 2-norm 
    for _ in range(5):
        gamma = 3 * (np.random.random() + 0.01)
        cost = lambda x, y: 0 if x[0] == y[0] else (2 + gamma)

        pnorm = skern.get_pnorm_regularized_runner(2., cost)

        epsilon = 1.
        delta = 0.0001

        ran = pnorm.run_sinkhorn(dist_a, dist_b, epsilon, delta)
        require(len(ran[0].elementMapping.keys()) == 4)
        require(sorted(list(ran[0].elementMapping.keys())) == [("H1", "H2"), ("H1", "T2"), ("T1", "H2"), ("T1", "T2")])
        
        pi = ran[0]
        require((pi.get_probability(("H1", "H2")) - 0.5) < delta)
        require((pi.get_probability(("T1", "T2")) - 0.5) < delta)
        require(pi.get_probability(("H1", "T2")) < delta)
        require(pi.get_probability(("T1", "H2")) < delta)

        f, g = ran[1], ran[2]
        alpha = f["H1"]
        beta = f["T1"]
        require(np.abs(g["H2"] - (2 - alpha)) < delta)
        require(np.abs(g["T2"] - (2 - beta)) < delta)
        require(np.abs(beta - alpha) <= gamma)

    for _ in range(5):
        eta = 2 * np.random.random()
        cost = lambda x, y: 0 if x[0] == y[0] else eta

        pnorm = skern.get_pnorm_regularized_runner(2., cost)

        epsilon = 1.
        delta = 0.0001

        ran = pnorm.run_sinkhorn(dist_a, dist_b, epsilon, delta)
        require(len(ran[0].elementMapping.keys()) == 4)
        require(sorted(list(ran[0].elementMapping.keys())) == [("H1", "H2"), ("H1", "T2"), ("T1", "H2"), ("T1", "T2")])
        
        pi = ran[0]
        require((pi.get_probability(("H1", "H2")) - (1 + eta / 2)) < delta)
        require((pi.get_probability(("T1", "T2")) - (1 + eta / 2)) < delta)
        require((pi.get_probability(("H1", "T2")) - (1 - eta / 2)) < delta)
        require((pi.get_probability(("T1", "H2")) - (1 - eta / 2)) < delta)

def run_all_tests():
    test_singleton()
    test_coin_flip_2norm()
    print("Ran all Tests for SinkhornRunnerTest")

if __name__ == "__main__":
    run_all_tests()


    


            


