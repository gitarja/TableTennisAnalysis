import pandas as pd
import numpy as np
from FeaturesReader import SingleFeaturesReader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from imblearn.under_sampling import CondensedNearestNeighbour
from sklearn.metrics import confusion_matrix
import xgboost
import shap
import matplotlib.pyplot as plt
from Validation.CrossValidation import SubjectCrossValidation
from Conf import single_results_path
from sklearn.model_selection import train_test_split
import pickle
import os
from imblearn.metrics import geometric_mean_score


os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'

#case we have a sequence of ACGT
np.random.seed(1945)

# prepare model for training

def trainXGB(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42)

    d_train = xgboost.DMatrix(X_train, label=y_train)
    d_test = xgboost.DMatrix(X_test, label=y_test)

    params = {
        "device": "cuda",
        "eta": 0.001,
        "objective": "reg:squarederror",
        "max_depth": 5,
        "eval_metric": "rmse",


    }
    model = xgboost.train(
        params,
        d_train,
        5000,
        evals=[(d_test, "test")],
        verbose_eval=False,
        early_stopping_rounds=20,

    )
    # fig, ax = plt.subplots(figsize=(30, 30))

    # plot_tree(model, ax=ax)
    # plt.tight_layout()
    # plt.show()
    return model


def evaluateModel(model, X_test, y_test):
    d_test = xgboost.DMatrix(X_test, label=y_test)

    y_pred = model.predict(d_test)
    predictions = [round(value) for value in y_pred]
    mse = mean_squared_error(y_test, predictions, squared=False)
    mae = mean_absolute_error(y_test, predictions)

    return mse, mae

mse_all, mae_all = [], []
shap_values_list, shap_interaction_list = [], []
X_test_list = []
X_test_display_list = []
# cnn = CondensedNearestNeighbour(random_state=42)
# load subject cross validation
features_reader = SubjectCrossValidation()
features_group = "all"
subject_train, subject_test = features_reader.getTrainTestData(1)
# load features
for i in range(len(subject_train)):
    path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\single_episode_features.pkl"

    train_subject = subject_train[i]
    test_subject = subject_test[i]
    features_reader_train = SingleFeaturesReader(path, include_subjects=train_subject)
    features_reader_test = SingleFeaturesReader(path, include_subjects=test_subject)

    X_train, y_train, x_column = features_reader_train.getIndividualObservationData(display=True, features_group=features_group)


    X_test, y_test, _ = features_reader_test.getIndividualObservationData(display=True, features_group=features_group)

    clf = trainXGB(X_train, y_train)

    mse, mae = evaluateModel(clf, X_test, y_test)

    # print("%.3f, %.3f" % (mse, mae))

    mse_all.append(mse)
    mae_all.append(mae)


    # compute shap
    X = np.concatenate([X_train, X_test])
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    shap_interaction_values = explainer.shap_interaction_values(X_test)

    # append test data
    X_test_list.append(X_test)
    # append results
    shap_values_list.append(shap_values)
    shap_interaction_list.append(shap_interaction_values)








print("%.3f, %.3f" % (np.average(mse_all), np.average(mae_all)))



# # save shap
X_test_all = pd.concat(X_test_list)
shap_values_all = np.vstack(shap_values_list)
shap_interaction_all = np.vstack(shap_interaction_list)
np.save(single_results_path + "x_gboost_reg_im_single_shap_values_"+features_group+".npy", shap_values_all)
np.save(single_results_path + "x_gboost_reg_im_single_shap_interaction_"+features_group+".npy", shap_interaction_all)
X_test_all.to_pickle(single_results_path + "x_gboost_reg_im_single_X_test_"+features_group+".pkl")

