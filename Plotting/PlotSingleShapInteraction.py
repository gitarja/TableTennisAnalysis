import numpy as np
from Conf import single_results_path
import pandas as pd
import shap
from Conf import x_perception, x_episode_columns
import matplotlib.pyplot as plt
import matplotlib
from Visualization.SHAPPlots import plotShapInteraction

np.random.seed(1945)


model = "gboost"

# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "arial"
results_path = single_results_path + "shap_values\\"
shap_values_all = np.load(single_results_path + "x_"+model+"_single_shap_values_all.npy")
X_test_all = pd.read_pickle(single_results_path + "x_"+model+"_single_X_test_all.pkl")

# only for LSTM
#shap_values_all = np.average(shap_values_all, axis=1)
# shap_values_all = shap_values_all[:, 2, :]
# X_test_all = X_test_all.loc[X_test_all["n_pos"] =="2", :x_episode_columns[-1]].apply(pd.to_numeric, errors='coerce')


plotShapInteraction(shap_values_all, X_test_all, x_episode_columns, results_path + "images\\", prefix=model, show_column= x_perception, ref="im_racket_force")


