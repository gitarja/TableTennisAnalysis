import numpy as np
from Conf import single_results_path
import pandas as pd
import shap
from Conf import x_episode_columns, x_important, x_perception
import matplotlib.pyplot as plt
import matplotlib
from Visualization.SHAPPlots import plotSHAP

np.random.seed(1945)


model = "lstm"
features_group = "all"
features = x_important
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "arial"
results_path = single_results_path + "shap_values\\"
shap_values_all = np.load(single_results_path + "x_"+model+"_single_shap_values_"+features_group+".npy")
X_test_all = pd.read_pickle(single_results_path + "x_"+model+"_single_X_test_"+features_group+".pkl")

# only for LSTM
#shap_values_all = np.average(shap_values_all, axis=1)
shap_list, x_test_list = [], []
for i in [2]:
    # shap_values_all = shap_values_all[:, 2, :]
    # X_test_all = X_test_all.loc[X_test_all["n_pos"] =="2", :features[-1]].apply(pd.to_numeric, errors='coerce')

    shap_list.append(shap_values_all[:,  i, :])
    x_test_list.append(X_test_all.loc[X_test_all["n_pos"] ==str(i), :features[-1]].apply(pd.to_numeric, errors='coerce'))

shap_values_all = np.vstack(shap_list)
X_test_all =  pd.concat(x_test_list)
# save mean
absolute_avg = np.mean(np.abs(shap_values_all), axis=0)
absolute_std = np.std(np.abs(shap_values_all), axis=0)
data = {"features_name": features, "Shap_ mean": absolute_avg, "Shap_ std": absolute_avg}

df = pd.DataFrame(data)

df.to_csv(single_results_path + "shap_"+model + "_" + features_group +"_2_values.csv")

# idc = np.array([features.index(c) for c in features])
#
# # shap.summary_plot(shap_values_all, X_test_all)
# shap.summary_plot(shap_values_all, X_test_all, plot_type="bar", max_display=len(features), show=False)
#
#
# plt.savefig(results_path + "images\\bar_"+model+"_lr.png", format='png')
# plt.close()
#
#
# plotSHAP(shap_values_all,X_test_all, features, results_path + "images\\", prefix=model)
