from finite_distributions.FiniteDistribution import FiniteDistribution
from core.require import require
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_joint_probability(joint_probability: FiniteDistribution, title = None, **kwargs):
    all_keys = joint_probability.elementMapping.keys()
    keys1 = sorted(list(set([tup[0] for tup in all_keys])))
    keys2 = sorted(list(set([tup[1] for tup in all_keys])))

    require(len(all_keys) == len(keys1) * len(keys2), f"Non-matching key lengths of {len(all_keys)} =!= {len(keys1)} * {len(keys2)}")

    dfCols = [{"key1": key1} | {key2: joint_probability.get_probability((key1, key2)) for key2 in keys2} for key1 in keys1]
    df = pd.DataFrame(dfCols).set_index("key1")

    plt.figure()
    sns.heatmap(df, **kwargs)
    plt.xlabel("key2")
    if title is not None:
        plt.title(title)
    plt.show()

