import core.core_testing as core_testing
import finite_distributions.finite_distribution_testing as finite_distribution_testing
import sinkhorn.sinkhorn_runner_test as sinkhorn_runner_test
import visualizer.visualizer_test as visualizer_test

if __name__ == "__main__":
    core_testing.run_all_tests()
    finite_distribution_testing.run_all_tests()
    sinkhorn_runner_test.run_all_tests()
    visualizer_test.run_all_tests()