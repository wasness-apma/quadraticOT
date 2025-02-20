
from finite_distributions.FiniteDistribution import FiniteDistribution
import visualizer.joint_distribution_visualizer as jdv

def joint_prob_vis_test_simple():
    joint = FiniteDistribution({("a1", "a2"): 0.1, ("a1", "b2"): 0.3, ("b1", "a2"): 0.6, ("b1", "b2"): 0.0})
    jdv.visualize_joint_probability(joint, annot = True)

def run_all_tests():
    print("Skipped visualizer tests.")

if __name__ == "__main__":
    joint_prob_vis_test_simple()