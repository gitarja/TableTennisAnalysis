import numpy as np
from Conf import double_results_path
import pandas as pd
import shap
from Conf import x_double_features_column
import matplotlib.pyplot as plt
import matplotlib
from Visualization.SHAPPlots import plotSHAP

np.random.seed(1945)

features = x_double_features_column

model = "gboost"
features_group = "all"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "arial"

results_path = double_results_path + "shap_values\\"
shap_values_all = np.load(results_path + "x_gboost_double_shap_values_all.npy")
# shap_interaction_all = np.load(results_path + "x_gboost_double_shap_interaction_all.npy")
X_test_all = pd.read_pickle(results_path + "x_gboost_double_X_test_all.pkl")

# save mean
absolute_avg = np.mean(np.abs(shap_values_all), axis=0)
absolute_std = np.std(np.abs(shap_values_all), axis=0)
data = {"features_name": x_double_features_column, "Shap_ mean": absolute_avg, "Shap_ std": absolute_avg}

df = pd.DataFrame(data)

df.to_csv(results_path + "shap_values.csv")

# plt.show()

shap.summary_plot(shap_values_all, X_test_all, max_display=len(features), alpha=0.1, show=False)

plt.savefig("F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\double\\shap_values\\images\\"+model+"_" + features_group+ "_"+"-summary-test.pdf", format="pdf", transparent=True)


