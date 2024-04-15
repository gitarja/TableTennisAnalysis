import pandas as pd
import numpy as np
from Single.FeaturesReader import SingleFeaturesReader
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, f1_score, roc_curve, auc
from imblearn.under_sampling import CondensedNearestNeighbour
from sklearn.metrics import confusion_matrix
import xgboost
from xgboost import plot_tree
import shap
import matplotlib.pyplot as plt
from Validation.CrossValidation import SubjectCrossValidation
from Conf import single_results_path
from sklearn.model_selection import train_test_split
import pickle
import os
from imblearn.metrics import geometric_mean_score
from Conf import  x_important

#case we have a sequence of ACGT
np.random.seed(1945)

# prepare model for training

def trainXGB(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42, stratify=y)

    d_train = xgboost.DMatrix(X_train, label=y_train)
    d_test = xgboost.DMatrix(X_test, label=y_test)

    params = {
        "device": "cuda",
        "eta": 0.001,
        "objective": "binary:logistic",
        "subsample": 0.5,
        # "base_score": np.mean(y_train),
        "max_depth": 5,
        "eval_metric": "logloss",
        "max_delta_step": 5,
        "alpha": 0.5,
        "scale_pos_weight": np.sum(y_train == 0 ) / (np.sum(y_train == 1 ) * 1.1)

    }
    model = xgboost.train(
        params,
        d_train,
        5000,
        evals=[(d_test, "test")],
        verbose_eval=False,
        early_stopping_rounds=20,

    )

    return model


def evaluateModel(model, X_test, y_test):
    d_test = xgboost.DMatrix(X_test, label=y_test)

    y_pred = model.predict(d_test)
    predictions = [round(value) for value in y_pred]
    mcc = matthews_corrcoef(y_test, predictions)
    cm = confusion_matrix(y_test, predictions, normalize="true")
    acc = balanced_accuracy_score(y_test, predictions)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    auc_score = auc(fpr, tpr)

    f1 = f1_score(y_test, predictions, average='weighted')

    g_mean = geometric_mean_score(y_test, predictions)

    print("ACC", acc)
    print("MCC", mcc)
    print("GMEAN", g_mean)
    print("-------------------------")
    return mcc, cm, acc, auc_score, f1, g_mean


def correlationCoeff(arr1, arr2):
    # Standardize the arrays
    arr1_standardized = (arr1 - np.mean(arr1)) / np.std(arr1)
    arr2_standardized = (arr2 - np.mean(arr2)) / np.std(arr2)

    # Compute the dot product
    dot_product = np.dot(arr1_standardized, arr2_standardized)

    # Calculate the correlation coefficient
    correlation = dot_product / len(arr1)

    return correlation

mcc_list, cm_list, acc_list, auc_list, f1_list, gmean_list = [], [], [], [], [], []

X_test_list = []
X_test_display_list = []
# cnn = CondensedNearestNeighbour(random_state=42)
# load subject cross validation
features_reader = SubjectCrossValidation()
features_group = "important"
subject_train, subject_test = features_reader.getTrainTestData(1)
number_of_bootstrap = 10
bootstrap_results = np.zeros((number_of_bootstrap * 5, len(x_important)))
index=0
# load features
for i in range(len(subject_train)):
    path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\ETRA2024_results_backup\\single_episode_features.pkl"

    train_subject = subject_train[i]
    test_subject = subject_test[i]
    features_reader_train = SingleFeaturesReader(path, include_subjects=train_subject)
    features_reader_test = SingleFeaturesReader(path, include_subjects=test_subject)

    X_train, y_train, x_column = features_reader_train.getIndividualObservationData(display=True, features_group=features_group)

    X_test, y_test, _ = features_reader_test.getIndividualObservationData(display=True, features_group=features_group)
    shap_values_list = []
    for j in range(number_of_bootstrap + 1):
        if j == 0:
            clf = trainXGB(X_train, y_train)
        else:
            bootstrap_indices = np.random.choice(X_train.shape[0], size=int(X_train.shape[0] * 0.75), replace=True)

            clf = trainXGB(X_train.iloc[bootstrap_indices, :], y_train[bootstrap_indices])

        # compute shap
        explainer = shap.explainers.GPUTreeExplainer(clf)
        shap_values = explainer.shap_values(X_test)
        # add shap values
        shap_values_list.append(shap_values)

    for k in range(1, number_of_bootstrap + 1):
        baseline = shap_values_list[0]
        comparison = shap_values_list[k]
        for l in range(len(x_column)):
           corr = correlationCoeff(baseline[:, l], comparison[:, l])
           bootstrap_results[index, l] = corr
        index +=1



np.savetxt(single_results_path + "confidence_shap_XGBoost_"+features_group + ".csv", bootstrap_results, delimiter=",")










