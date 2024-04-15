import pandas as pd
from Conf import single_results_path, results_path
from Conf import x_episode_columns
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
from scipy import stats


custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)



X = pd.read_pickle(results_path + "single_episode_features.pkl")
corr_matrix = np.zeros((len(x_episode_columns), len(x_episode_columns)))
for i in range(len(x_episode_columns)):
    for j in range(len(x_episode_columns)):
        f1 = x_episode_columns[i]
        f2 = x_episode_columns[j]
        x1 = X.loc[:, f1].values
        x2 = X.loc[:, f2].values

        mask = ~ (np.isnan(x1) | np.isnan(x2))

        #remove nan
        corr = stats.spearmanr(x1[mask], x2[mask])

        if i != j:
            corr_matrix[i, j] = corr[1]




df = pd.DataFrame(corr_matrix, columns=x_episode_columns)

df.to_csv(single_results_path + "summary_correlation_features.csv")