import pandas as pd
from Conf import single_results_path, results_path, double_results_path, double_summary_path, single_summary_path
from Conf import x_double_features_all_column
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

sns.set_theme()

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)


df_summary = pd.read_csv(single_summary_path)
df_summary = df_summary[(df_summary["Tobii_percentage"] > 65)]

df = pd.read_pickle(results_path + "single_episode_features.pkl")


# df = df.loc[(df["team_skill"]>0.55) &(df["success"]!=-1) & (df["session_id"].isin(df_summary["file_name"].values)) & (df["session_id"]!="2022-12-19_A_T06")]
# df_filtered = df[(df["pair_idx"] > 0) & (df["pair_idx"] < 45)]

df = df.loc[
    (df["skill_subject"] > 0.55) & (df["success"] != -1) & (df["id_subject"].isin(df_summary["Subject1"].values))]


# features_group_pair = df.groupby('pair_idx')
df_filtered = df[(df["observation_label"] < 50)]

sns.lineplot(
    data=df_filtered, x="observation_label", y="pr_p1_al_miDo",

    hue="success"

)





plt.show()

