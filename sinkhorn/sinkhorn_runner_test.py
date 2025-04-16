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

def test_coin_flip_quadratic_runner():
    # test comes from Nutz paper, "Quadratically Regularized Optima Transport: Exitence and Multiplicity of Potentials"
    dist_a_keys = {"H1": 0.5, "T1": 0.5}
    dist_a = FiniteDistribution(dist_a_keys)

    dist_b_keys = {"H2": 0.5, "T2": 0.5}
    dist_b = FiniteDistribution(dist_b_keys)

    # test cost for 2-norm 
    for _ in range(5):
        gamma = 3 * (np.random.random() + 0.01)
        cost = lambda x, y: 0 if x[0] == y[0] else (2 + gamma)

        two_norm = skern.get_quadratically_regularized_runner(cost, use_parallelization=False)

        epsilon = 1.
        delta = 0.0001

        ran = two_norm.run_sinkhorn(dist_a, dist_b, epsilon, delta, printInfo = False)
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


def test_compare_quadratic_kernels():
    for use_par in [False]:
        for nkeys in [2]:#range(1, 10):
            # scaling
            dist_a_keys_unsummed = {i: 1.0 * (i+1) for i in range(nkeys)}
            sum_a = sum([v for v in dist_a_keys_unsummed.values()])
            dist_a_keys = {key: (dist_a_keys_unsummed[key])/sum_a for key in dist_a_keys_unsummed}
            dist_a = FiniteDistribution(dist_a_keys)

            # uniform
            dist_b_keys_unsummed = {i: 1.0 for i in range(nkeys)}
            sum_b = sum([v for v in dist_b_keys_unsummed.values()])
            dist_b_keys = {key: dist_b_keys_unsummed[key]/sum_b for key in dist_b_keys_unsummed}
            dist_b = FiniteDistribution(dist_b_keys)

            cost = lambda x, y: (x - y)**2

            runner_standard = skern.get_pnorm_regularized_runner(2.0, cost)
            runner_optimized = skern.get_quadratically_regularized_runner(cost, use_parallelization=use_par)

            epsilon = 1.0
            precisionDelta = 0.01
            pi_s, f_s, g_s, _, _ = runner_standard.run_sinkhorn(dist_a, dist_b, epsilon, precisionDelta=precisionDelta)
            pi_q, f_q, g_q, _, _ = runner_optimized.run_sinkhorn(dist_a, dist_b, epsilon, precisionDelta=precisionDelta)

            require(pi_s.get_keys() == pi_q.get_keys())
            require(f_s.keys() == f_q.keys())
            require(g_s.keys() == g_q.keys())

            for (x, y) in pi_s.get_keys():
                requireApproxEq(pi_s.get_probability((x, y)), pi_q.get_probability((x, y)), epsilon=precisionDelta)
                requireApproxEq(f_s[x], f_q[x], epsilon=precisionDelta)
                requireApproxEq(g_s[y], g_q[y], epsilon=precisionDelta)

def run_all_tests():
    test_singleton()
    test_coin_flip_2norm()
    test_coin_flip_quadratic_runner()
    test_compare_quadratic_kernels()
    print("Ran all Tests for SinkhornRunnerTest")

if __name__ == "__main__":
    run_all_tests()


    


            


