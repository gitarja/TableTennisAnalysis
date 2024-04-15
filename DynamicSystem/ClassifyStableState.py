import pandas as pd
from Conf import single_results_path, results_path, double_results_path, double_summary_path, single_summary_path
from pomegranate.distributions import Normal
from pomegranate.hmm import DenseHMM, SparseHMM
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from Conf import x_double_features_column, normalize_x_double_episode_columns
from sklearn.impute import KNNImputer
torch.random.manual_seed(1945)
import seaborn as sns
MODEL_PATH = "..\\CheckPoints\\HMM\\"

def cohend(d1, d2):
 # calculate the size of samples
 n1, n2 = len(d1), len(d2)
 # calculate the variance of the samples
 s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
 # calculate the pooled standard deviation
 s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
 # calculate the means of the samples
 u1, u2 = np.mean(d1), np.mean(d2)
 # calculate the effect size
 return (u1 - u2) / s

def createAndTrainModel(X):


    d1 = Normal()
    d2 = Normal()


    model = SparseHMM(verbose=True, tol=0.01)
    model.add_distributions([d1, d2])
    # model.add_edge(model.start, d1, 0.5)
    model.add_edge(model.start, d2, 0.5)
    model.add_edge(d1, d1, 0.5)
    model.add_edge(d1, d2, 0.5)
    model.add_edge(d2, d1, 0.5)
    model.add_edge(d2, d2, 0.5)
    model.add_edge(d2, model.end, 0.9)
    # model.add_edge(d2, model.end, 0.5)

    model.fit(X)

    torch.save(model, MODEL_PATH + "double_hmm.pth")
    return model


df_summary = pd.read_csv(double_summary_path)
df_summary = df_summary[(df_summary["norm_score"] > 0.55) & (df_summary["Tobii_percentage"] > 65)]

df = pd.read_pickle(results_path + "double_episode_features.pkl")


df = df.loc[(df["success"] != -1) & (df["pair_idx"] != -1) &(df["session_id"].isin(df_summary["file_name"].values)), :]

df = df.loc[(df["session_id"] != "2022-12-19_A_T06") | (
        df["session_id"] != "2023-02-15_M_T01")]  # session excluded, equipments fail

# normalize features
mean = np.nanmean(
    df.loc[:, normalize_x_double_episode_columns], axis=0)
std = np.nanstd(
    df.loc[:, normalize_x_double_episode_columns], axis=0)
df.loc[:, normalize_x_double_episode_columns] = (df.loc[:,
                                                 normalize_x_double_episode_columns] - mean) / std

# input missing values
imputer = KNNImputer(n_neighbors=5)
df.loc[:, normalize_x_double_episode_columns] = imputer.fit_transform(df.loc[:, normalize_x_double_episode_columns])

selected_groups = df.groupby(["session_id", "episode_label"]).filter(lambda x: len(x) >= 5)

grouped_episodes = selected_groups.groupby(["session_id", "episode_label"])

X = []
X_test = []

for i, g in grouped_episodes:



    print(g["session_id"].values[0])
    # Generate a random signal
    g = g.sort_values(by=['observation_label'])
    # g = g.loc[g["receiver"].values==1]
    signal1 = g["hitter_hit_to_bouncing_point"].values
    signal2 = g["receiver_im_ball_force"].values
    signal3 = g["receiver_im_racket_force"].values
    signal4 = g["receiver_racket_to_root"].values
    signal5 = g["receiver_ec_fs_ball_racket_dir"].values
    signal6 = g["receiver_ec_start_fs"].values

    test = g["receiver_racket_to_root"].values

    if len(signal1) > 0:
        X.append(np.vstack([signal2, signal3,  signal5]).T)

        X_test.append(test)


# model = createAndTrainModel(X)
model = torch.load(MODEL_PATH + "best2_double_hmm.pth")
stable_features = []
semi_stable_features = []
for i in range(len(X)):
    x_test = np.expand_dims(X[i], axis=0)
    probs = model.predict_proba(x_test).numpy()[0]

    mask = probs[:, 0] > 0.5
    preds = model.predict(x_test).numpy()[0]



    # stable_features.append(X_test[i][mask == True])
    # semi_stable_features.append(X_test[i][mask == False])


    s_to_ss = np.pad(np.diff(preds), (1, 0), 'edge')

    stable_features.append(X_test[i][(s_to_ss == 0) & (preds == 0)])
    semi_stable_features.append(X_test[i][(s_to_ss == 1) | ((s_to_ss == 0) & (preds == 1))])






    # # # visualize
    # # print(preds)
    # fig, axs = plt.subplots(4)
    # marks = np.argwhere(preds.flatten()==1)
    #
    # axs[0].plot(x_test[0, :, 0], color="#252525")
    # ax0_1 = axs[0].twinx()
    # ax0_1.scatter(np.argwhere(preds==1), np.ones_like(np.argwhere(preds==1)), color="#f57f20", marker='o')
    # ax0_1.scatter(np.argwhere(preds == 0), np.zeros_like(np.argwhere(preds == 0)), color="#4eaf49", marker='o')
    #
    # axs[1].plot(x_test[0, :, 1], color="#252525")
    # ax0_1 = axs[1].twinx()
    # ax0_1.scatter(np.argwhere(preds == 1), np.ones_like(np.argwhere(preds == 1)), color="#f57f20", marker='o')
    # ax0_1.scatter(np.argwhere(preds == 0), np.zeros_like(np.argwhere(preds == 0)), color="#4eaf49", marker='o')
    #
    #
    # axs[2].plot(x_test[0, :, 2], color="#252525")
    # ax0_1 = axs[2].twinx()
    # ax0_1.scatter(np.argwhere(preds == 1), np.ones_like(np.argwhere(preds == 1)), color="#f57f20", marker='o')
    # ax0_1.scatter(np.argwhere(preds == 0), np.zeros_like(np.argwhere(preds == 0)), color="#4eaf49", marker='o')
    #
    #
    # axs[3].plot(X_test[i], color="#252525")
    # ax1_1 = axs[3].twinx()
    # ax1_1.scatter(np.argwhere(preds==1), np.ones_like(np.argwhere(preds==1)), color="#f57f20", marker='o')
    # ax1_1.scatter(np.argwhere(preds == 0), np.zeros_like(np.argwhere(preds == 0)), color="#4eaf49", marker='o')
    #
    # plt.show()


#plot
from scipy import stats
s = np.concatenate(stable_features)
ss = np.concatenate(semi_stable_features)
print(stats.ttest_ind(s[~ np.isnan(s)], ss[~ np.isnan(ss)]))
print(cohend(s[~ np.isnan(s)], ss[~ np.isnan(ss)]))
data = np.concatenate([np.concatenate(stable_features), np.concatenate(semi_stable_features)])

label = np.concatenate([["s" for i in range(len(np.concatenate(stable_features)))], ["ss" for i in range(len(np.concatenate(semi_stable_features)))]])
df = pd.DataFrame(data={"value": data, "state":label})
# sns.displot(df, x="x", hue="y", kind="kde")
g = sns.catplot(
    data=df,  y="value", hue="state",
    kind="box"
)
# g.set(ylim=(0, 20))
# sns.stripplot(data=df, x="state", y="x",   ax=g)
# sns.swarmplot(data=df, hue="state", y="x", color="k", size=0.5, ax=g.ax)

plt.show()