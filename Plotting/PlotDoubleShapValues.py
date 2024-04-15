import numpy as np
from Conf import double_results_path
import pandas as pd
import shap
from Conf import x_double_features_column
import matplotlib.pyplot as plt
import matplotlib
from Visualization.SHAPPlots import plotSHAP

np.random.seed(1945)


model = "gboost"
features_group = "all"
features = x_double_features_column
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "arial"
results_path = double_results_path + "shap_values\\"
shap_values_all = np.load(results_path + "x_"+model+"_double_shap_values_"+features_group+".npy")
X_test_all = pd.read_pickle(results_path + "x_"+model+"_double_X_test_"+features_group+".pkl")


# save mean
absolute_avg = np.mean(np.abs(shap_values_all), axis=0)
absolute_std = np.std(np.abs(shap_values_all), axis=0)
data = {"features_name": features, "Shap_ mean": absolute_avg, "Shap_ std": absolute_avg}

df = pd.DataFrame(data)

df.to_csv(double_results_path + "shap_"+model + "_" + features_group +"_2_values.csv")

idc = np.array([features.index(c) for c in features])

# shap.summary_plot(shap_values_all, X_test_all)
shap.summary_plot(shap_values_all, X_test_all, plot_type="bar", max_display=len(features), show=False)


plt.savefig(results_path + "images\\bar_"+model+"_lr.png", format='png')
plt.close()


plotSHAP(shap_values_all,X_test_all, features, results_path + "images\\", prefix=model)
