import pandas as pd
import numpy as np
from FeatureReader import DoubleFeaturesReader
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, f1_score, roc_curve, auc

from sklearn.metrics import confusion_matrix
import xgboost
from xgboost import plot_tree
import shap
import matplotlib.pyplot as plt
from Validation.CrossValidation import DoubleSubjectCrossValidation
from Conf import double_results_path
from sklearn.model_selection import train_test_split
import pickle
import os
from imblearn.metrics import geometric_mean_score
from corr_shap import CorrExplainer
from shap.utils._legacy import LogitLink
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'

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
        "max_delta_step": 5,
        "max_depth": 5,
        "eval_metric": "logloss",
        "alpha":0.5,
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
    # fig, ax = plt.subplots(figsize=(30, 30))

    # plot_tree(model, ax=ax)
    # plt.tight_layout()
    # plt.show()
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
    print("-------------------------")
    return mcc, cm, acc, auc_score, f1, g_mean

mcc_list, cm_list, acc_list, auc_list, f1_list, gmean_list = [], [], [], [], [], []
shap_values_list, shap_interaction_list = [], []
X_test_list = []
X_test_display_list = []
# load subject cross validation
features_reader = DoubleSubjectCrossValidation()
features_group = "all"
subject_train, subject_test = features_reader.getTrainTestData(1)
# load features
for i in range(len(subject_train)):
    path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\double_episode_features.pkl"

    train_subject = subject_train[i]
    test_subject = subject_test[i]
    features_reader_train = DoubleFeaturesReader(path, include_subjects=train_subject)
    features_reader_test = DoubleFeaturesReader(path, include_subjects=test_subject)

    X_train, y_train, x_column = features_reader_train.getIndividualObservationData(display=True, features_group=features_group)

    X_test, y_test, _ = features_reader_test.getIndividualObservationData(display=True, features_group=features_group)

    X_test_disp, _, _ = features_reader_test.getIndividualObservationData(display=True, features_group=features_group)




    clf = trainXGB(X_train, y_train)

    mcc, cm, acc, auc_score, f1, g_mean = evaluateModel(clf, X_test, y_test)

    mcc_list.append(mcc)
    cm_list.append(cm)
    acc_list.append(acc)
    f1_list.append(f1)
    auc_list.append(auc_score)
    gmean_list.append(g_mean)

    # # compute shap
    # X = np.concatenate([X_train, X_test])
    # explainer = shap.TreeExplainer(clf, data=X_train, feature_perturbation="interventional")
    # shap_values = explainer.shap_values(X_test)



    X_background = shap.kmeans(X_train.values, k=15).data
    # clf.set_param({"device": "cuda:0"})
    explainer = CorrExplainer(clf.inplace_predict, X_background, sampling="gauss+empirical", link=LogitLink())
    shap_values = explainer.shap_values(X_test.values)

    # append test data
    X_test_list.append(X_test_disp)
    # append results
    shap_values_list.append(shap_values)






mcc_all = np.vstack(mcc_list)
cm_all = np.array(cm_list)
acc_all = np.vstack(acc_list)
f1_all = np.array(f1_list)
auc_all = np.array(auc_list)
gmean_all = np.array(gmean_list)

print(np.average(acc_all))
print(np.average(mcc_all))
print(np.average(gmean_all))


# # # save shap
X_test_all = pd.concat(X_test_list)
shap_values_all = np.vstack(shap_values_list)
results_path  = double_results_path + "shap_values\\"
np.save(results_path  + "x_gboost_double_shap_values_"+features_group+".npy", shap_values_all)
X_test_all.to_pickle(results_path + "x_gboost_double_X_test_"+features_group+".pkl")
#
np.save(results_path + "x_gboost_double_MCC_"+features_group+".npy", mcc_all)
np.save(results_path + "x_gboost_double_confusion_matrix_"+features_group+".npy", cm_all)
np.save(results_path + "x_gboost_double_acc_"+features_group+".npy", acc_all)
np.save(results_path + "x_gboost_double_f1_"+features_group+".npy", f1_all)
np.save(results_path + "x_gboost_double_auc_"+features_group+".npy", auc_all)


