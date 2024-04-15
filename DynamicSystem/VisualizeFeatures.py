import pandas as pd
from Conf import single_results_path, results_path, double_results_path, double_summary_path
import numpy as np
from pomegranate.distributions import Normal, Categorical
from pomegranate.hmm import DenseHMM
import torch
from Lib import computePowerSpectrum, movingAverage
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler





df_summary = pd.read_csv(double_summary_path)
df_summary = df_summary[(df_summary["norm_score"] > 0.55) & (df_summary["Tobii_percentage"] > 65)]

df = pd.read_pickle(results_path + "double_episode_features.pkl")


df = df.loc[(df["success"] != -1) & (df["pair_idx"] != -1) &(df["session_id"].isin(df_summary["file_name"].values)), :]

df = df.loc[(df["session_id"] != "2022-12-19_A_T06") | (
        df["session_id"] != "2023-02-15_M_T01")]  # session excluded, equipments fail

selected_groups = df.groupby(["session_id", "episode_label"]).filter(lambda x: len(x) > 20)

grouped_episodes = selected_groups.groupby(["session_id", "episode_label"])


scaller = StandardScaler()
for i, g in grouped_episodes:



    print(g["session_id"].values[0])
    # Generate a random signal
    g = g.sort_values(by=['observation_label'])
    # g = g.loc[g["receiver"].values==1]
    signal1 = movingAverage(g["hitter_hit_to_bouncing_point"].values, 1)
    signal2 = movingAverage(g["receiver_im_ball_force"].values, 1)
    signal3 = movingAverage(g["receiver_im_rb_ang_collision"].values, 1)
    signal4 = movingAverage(g["receiver_pr_p1_al_prec"].values, 1)

    # change point detection

    d1 = Normal()
    d2 = Normal()

    edges = [[0.5, 0.5], [0.5, 0.5]]
    starts = [0.5, 0.5]
    ends = [0.5, 0.5]
    X = np.expand_dims(np.vstack([signal1, signal2, signal3]).T, 0)
    model = DenseHMM([d1, d2], edges=edges, starts=starts, ends=ends, verbose=True, tol=0.001)
    model.fit(X)
    preds = model.predict(X)
    marks = np.argwhere(preds==1)[1]
    fig, axs = plt.subplots(4)

    axs[0].plot(signal1)
    axs[0].vlines(x=marks, ymin=0, ymax=np.max(signal1), colors='purple')
    axs[1].plot(signal2)
    axs[1].vlines(x=marks, ymin=0, ymax=np.max(signal2), colors='purple')
    axs[2].plot(signal3)
    axs[2].vlines(x=marks, ymin=0, ymax=np.max(signal3), colors='purple')
    axs[3].plot(signal4)
    axs[3].vlines(x=marks, ymin=0, ymax=np.max(signal4), colors='purple')
    plt.show()

    # model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
    # n = len(signal1) - 0
    # sigma = 0.1
    #
    #
    #
    # fig, axs = plt.subplots(2)
    # algo = rpt.Window(width=3, model=model).fit(signal1)
    # my_bkps1 = algo.predict(epsilon= n * 1e-10 ** 2)
    # axs[0].plot(signal1)
    # axs[0].vlines(x=my_bkps1, ymin=0, ymax=np.max(signal1), colors='purple')
    #
    # algo = rpt.Window(width=3, model=model).fit(signal2)
    # my_bkps2 = algo.predict(epsilon=n * 1e-10 ** 2)
    # axs[1].plot(signal2)
    # axs[1].vlines(x=my_bkps2, ymin=0, ymax=np.max(signal2), colors='purple')
    #
    #
    # plt.show()
