import numpy as np
import matplotlib.pyplot as plt
from DynamicSystem.DataReader import SingleFeatures, DoubleDataReader
from Conf import  results_path
import nolds
import pandas as pd
import seaborn as sns
from Lib import movingAverage

# get single group
single_features_path = results_path + "single_episode_features.pkl"
single_reader = SingleFeatures(single_features_path)
single_group_episodes = single_reader.getGroup(min_n=20)
single_features_name = "im_racket_force"


# get pair group
double_features_path = results_path + "double_episode_features.pkl"
double_reader = DoubleDataReader(double_features_path)
double_group_episodes = double_reader.getGroup(min_n=20)
double_features_name = "receiver_im_racket_force"

matrix_dim = 3
embd_dim = 9

# compute lypanov of single
single_exponent = []
single_seq_n = []
single_exponent_label = []
for i, g in single_group_episodes:
    g = g.sort_values(by=['observation_label'])
    x = movingAverage(g[single_features_name].values, 3)

    le = nolds.lyap_e(x,  emb_dim=embd_dim, matrix_dim=matrix_dim)

    for j in range(len(le)):
        single_seq_n.append(len(x))
        single_exponent.append(le[j])
        single_exponent_label.append(j)

# single_exponent = np.asarray(single_exponent)
# single_exponent_df = pd.DataFrame(data={"exponent": single_exponent, "label":single_seq_n, "exponent_label":single_exponent_label })
# g = sns.lineplot(data=single_exponent_df, x="label", y="exponent", hue="exponent_label", palette=sns.color_palette("crest", as_cmap=True), estimator=np.median)
# g.hlines(y=0,  xmin=np.min(single_seq_n), xmax=np.max(single_seq_n),  ls='--', lw=1)
# sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
# plt.show()


# compute lypanov of double
double_exponent = []
double_seq_n = []
double_exponent_label = []
for i, g in double_group_episodes:
    g = g.sort_values(by=['observation_label'])

    # first subject
    g0 = g.loc[g["receiver"].values == 0]
    x0 = movingAverage(g0[double_features_name].values, 3)

    # second subject
    g1 =  g.loc[g["receiver"].values == 1]
    x1 = movingAverage(g1[double_features_name].values, 3)



    le0 = nolds.lyap_e(x0,   emb_dim=embd_dim, matrix_dim=matrix_dim)
    le1 = nolds.lyap_e(x1, emb_dim=embd_dim, matrix_dim=matrix_dim)

    for j in range(len(le0)):
        double_exponent.append(le0[j])
        double_exponent.append(le1[j])
        double_seq_n.append(len(x0))
        double_seq_n.append(len(x1))
        double_exponent_label.append(j)
        double_exponent_label.append(j)


double_exponent = np.asarray(double_exponent)
single_exponent_df = pd.DataFrame(data={"exponent": double_exponent, "label":double_seq_n, "exponent_label":double_exponent_label })
g = sns.lineplot(data=single_exponent_df, x="label", y="exponent", hue="exponent_label", palette=sns.color_palette("crest", as_cmap=True), estimator=np.median)
g.hlines(y=0,  xmin=np.min(double_seq_n), xmax=np.max(double_seq_n),  ls='--', lw=1)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
plt.show()