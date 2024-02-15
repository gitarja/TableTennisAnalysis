import pandas as pd
from Conf import single_results_path, results_path
from Conf import x_statistic_important, x_episode_columns
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

X = pd.read_pickle(results_path + "single_episode_features.pkl")

# data_df = []
#
# for f1 in x_episode_columns:
#     for f2 in x_episode_columns:
#         XS = X.dropna(subset=[f1, f2])
#         # sns.regplot(data=X, x=f1, y=f2, scatter_kws=dict(s=5), line_kws=dict(color="r"))
#
#
#         pearson_score = pearsonr(XS[f1].values, XS[f2].values)
#
#         data_df.append([f1, f2, pearson_score.correlation, len(XS) - 2, pearson_score.pvalue])
#
# df = pd.DataFrame(data_df, columns=['f1', 'f2', "correlation", "N", "p-val"])
#
# df.to_csv(single_results_path + "summary_correlation.csv")
#

f1 = 'pr_p1_al_miDo'
f2 = 'ec_start_fs'
X = X.dropna(subset=[f1, f2])
sns.regplot(data=X, x=f1, y=f2, scatter_kws=dict(s=5), line_kws=dict(color="r"))
pearson_score = pearsonr(X[f1].values, X[f2].values)
print(pearson_score)
# plt.show()
plt.tight_layout()
plt.savefig(single_results_path + "\\correlation\\correlation_"+ f1 + "_" + f2 + '.eps', format='eps')
