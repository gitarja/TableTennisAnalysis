import pandas as pd
from Conf import single_results_path, results_path
from Conf import x_statistic_important, x_episode_columns
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)


def cohenD(a, b):
    '''
    Compute cohen coefficient of paired data
    :param a: 1st data
    :param b: 2nd data
    :return: coefficient
    '''
    n_a = len(a)
    n_b = len(b)
    return (np.mean(a)-np.mean(b))/\
           np.sqrt(((n_a-1)*np.var(a, ddof=1) + (n_b-1)*np.var(b, ddof=1)) / (n_a + n_b -2))


X = pd.read_pickle(results_path + "single_episode_features.pkl")

data_df = []
for f1 in x_episode_columns:
    X = X.dropna(subset=[f1])

    a = X.loc[X["success"] == 1, f1]
    b = X.loc[X["success"] == 0, f1]

    d_score = cohenD(a, b)

    mean_s = np.mean(a)
    std_s = np.std(a)

    mean_f = np.mean(b)
    std_f = np.std(b)

    data_df.append([f1, d_score, mean_s, std_s, mean_f, std_f])

df = pd.DataFrame(data_df, columns=['f1', "d_score", "mean_s", "std_s", "mean_f", "std_f"])

df.to_csv(single_results_path + "summary_cohen_avg_std.csv")