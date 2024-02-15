import numpy as np
from Conf import single_results_path
import matplotlib.pyplot as plt
from Conf import x_important, x_episode_columns
import seaborn as sns

idc = np.array([x_episode_columns.index(c) for c in x_episode_columns])

shap_interaction_all = np.load(single_results_path + "x_gboost_single_shap_interaction_all.npy")

shap_interaction = np.mean(np.abs(shap_interaction_all), axis=0)
shap_interaction = shap_interaction[idc]
shap_interaction = shap_interaction[:, idc]
shap_interaction = np.fill_diagonal(shap_interaction, 0)
columns = np.asarray(x_important)

# for i in range(shap_interaction.shape[0]):
#     shap_interaction[i, i] = 0
# inds = np.argsort(-shap_interaction.sum(0))
# tmp2 = shap_interaction[inds, :][:, inds]
fig = plt.figure(figsize=(35, 20))
ax = fig.add_subplot()
sns.heatmap(np.round(shap_interaction, decimals=1),  cmap='coolwarm', annot=True, fmt='.6g', cbar=False,)
# plt.yticks(
#     np.arange(shap_interaction.shape[0])+ 0.5, columns, rotation=0, horizontalalignment="right"
# )
# plt.xticks(
#     np.arange(shap_interaction.shape[0])+ 0.5, columns, rotation=50, horizontalalignment="left"
# )
# plt.gca().xaxis.tick_top()
plt.show()