import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from joblib import Parallel, delayed
import more_itertools

from nonword_filters import *

COSINE_RADIUS_RANGE = [0.55, 0.525, 0.5, 0.475, 0.45, 0.425, 0.4, 0.375, 0.35]

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

def parallelize_func(iterable, func, chunksz = 1, n_jobs = 16, *args, **kwargs):
    chunker = func
    chunks = more_itertools.chunked(iterable, chunksz)
    chunks_results = Parallel(n_jobs=n_jobs, verbose = 50)(delayed(chunker)(chunk, *args, **kwargs) for chunk in chunks)
    results = more_itertools.flatten(chunks_results) # type: ignore
    return list(results)


def plot_neighbourhood_stats(df_statistic, statistic, outfile):
    """
    Visualizing a bar chart of mean neighbourhood density or frequency growth rate
    :param mean_neologism_statistic: a list of values of the statistic for different neighbourhood sizes,
    averaged over all neologisms
    :param mean_control_statistic: a list of values of the statistic for different neighbourhood sizes,
    averaged over all control words
    :param statistic: type of the statistic provided ('density', 'growth monotonicity', 'growth slope', 'hashtag monotonicity', 'hashtag slope')
    :return:
    """

    sns.set_theme()

    assert statistic in ["density", "growth monotonicity", "growth slope"]
    column_name = "".join([w.capitalize() for w in statistic.split()])

    sns.set_theme()
    fig, ax = plt.subplots()

    df_statistic["Neighbourhoods of:"] = df_statistic["IsNeologism"].map({0: 'Control words', 
                                                                        1: 'Neologisms'})
    df_plot = pd.melt(df_statistic, id_vars = ["Neighbourhoods of:"], value_vars = 
                                        [f"{column_name}AtRadius{r:.3f}" for r in COSINE_RADIUS_RANGE])
    sns.barplot(data=df_plot.dropna(), x = "variable", y = "value", hue = "Neighbourhoods of:", 
                            errorbar=('ci', 95))
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([l.get_text()[-5:] for l in ax.get_xticklabels()])
    ax.set_xlabel("Neighbourhood cosine distance threshold")

    if statistic == "density":
        ax.set_title("Average number of neighbour words in radius")
        ax.set_ylabel("Neighbourhood density")
    else:
        ax.set_title(f"Average frequency {statistic} of the neighbour words")
        ax.set_ylabel(f"Neighbourhood frequency {statistic}")

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300)
    else:
        plt.show()
