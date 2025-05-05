from finite_distributions.FiniteDistribution import FiniteDistribution
from core.require import require
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_joint_probability(joint_probability: FiniteDistribution, title = None, support_only_threshold = None, xticks = None, yticks = None, show = False, ylabel_to_set = None, xlabel_to_set = None, reverse_sort_y_axis = False, **kwargs):
    all_keys = joint_probability.elementMapping.keys()
    keys1 = sorted(list(set([tup[0] for tup in all_keys])))
    keys2 = sorted(list(set([tup[1] for tup in all_keys])))
    if (reverse_sort_y_axis):
        keys2 = list(reversed(keys2))

    require(len(all_keys) == len(keys1) * len(keys2), f"Non-matching key lengths of {len(all_keys)} =!= {len(keys1)} * {len(keys2)}")

    if support_only_threshold is None:
        dfCols = [{"key1": key1} | {key2: joint_probability.get_probability((key1, key2)) for key2 in keys2} for key1 in keys1]
    else:
        dfCols = [{"key1": key1} | {
                key2: (1.0 if joint_probability.get_probability((key1, key2)) >= support_only_threshold else 0.0)
                for key2 in keys2
            } for key1 in keys1]

    df = pd.DataFrame(dfCols).set_index("key1")

    if show:
        plt.figure()
    ax = sns.heatmap(df, **kwargs)
    plt.xlabel("key2")
    if ylabel_to_set is not None:
        ax.set_ylabel(ylabel_to_set)
    if xlabel_to_set is not None:
        ax.set_xlabel(xlabel_to_set)
    if title is not None:
        plt.title(title)
    if xticks is not None:
        ax.set_xticks(xticks[0])
        ax.set_xticklabels(xticks[1])
    if yticks is not None:
        ax.set_yticks(yticks[0])
        ax.set_yticklabels(yticks[1])
    if show:
        plt.show()

