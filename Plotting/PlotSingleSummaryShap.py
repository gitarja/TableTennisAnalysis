import numpy as np
from Conf import single_results_path
import pandas as pd
import shap
from Conf import x_episode_columns, x_important, x_perception
import matplotlib.pyplot as plt
import matplotlib
import shap

np.random.seed(1945)


model = "LSTM"
features = x_important
features_group="all"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "arial"
results_path = single_results_path + "shap_values\\"
shap_values_all = np.load(single_results_path + "x_"+model+"_single_shap_values_"+features_group+".npy")
X_test_all = pd.read_pickle(single_results_path + "x_"+model+"_single_X_test_"+features_group+".pkl")

# only for LSTM
shap_list, x_test_list = [], []
for i in [2]:
    shap_list.append(shap_values_all[:,  i, :])
    x_test_list.append(X_test_all.loc[X_test_all["n_pos"] ==str(i), :features[-1]].apply(pd.to_numeric, errors='coerce'))
shap_values_all = np.vstack(shap_list)
X_test_all =  pd.concat(x_test_list)

## end only for LSTM


shap.summary_plot(shap_values_all, X_test_all, max_display=len(features), alpha=0.1, show=False)

plt.savefig("F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\single\\shap_values\\images\\LSTM_2-important-summary.pdf", format="pdf", transparent=True)

