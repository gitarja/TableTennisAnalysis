import pandas as pd
from Conf import single_results_path, results_path, double_results_path, double_summary_path
from Conf import x_double_features_all_column,y_episode_column
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn import preprocessing

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)


df_summary = pd.read_csv(double_summary_path)
df_summary = df_summary[(df_summary["Tobii_percentage"] > 65)]

df = pd.read_pickle(results_path + "double_episode_features.pkl")


df = df.loc[(df["team_skill"]>0.55) &(df["success"]==0) & (df["session_id"].isin(df_summary["file_name"].values)) & (df["session_id"]!="2022-12-19_A_T06")]


features_df = df.loc[:, x_double_features_all_column]

# mean and team-skill
# features_mean = features_df.groupby('session_id').mean()

# for i in range(2, len(x_double_features_all_column)):
#     g = sns.jointplot(data=features_mean, x=x_double_features_all_column[i], y="team_skill")
#     g.plot_joint(sns.regplot)
#     plt.savefig(double_results_path + x_double_features_all_column[i] + ".png")
#     plt.close()


for i in range(2, len(x_double_features_all_column)):
    g = sns.jointplot(data=features_df, x=x_double_features_all_column[i], y="lcc_p23_sim",  kind="hex")
    plt.savefig(double_results_path + x_double_features_all_column[i] + "_detail.png")
    plt.close()