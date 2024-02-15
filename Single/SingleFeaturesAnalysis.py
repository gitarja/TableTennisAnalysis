import pandas as pd
import numpy as np
from FeaturesReader import SingleFeaturesReader
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
    print("GMEAN", g_mean)
    print("-------------------------")
    return mcc, cm, acc, auc_score, f1, g_mean

mcc_list, cm_list, acc_list, auc_list, f1_list, gmean_list = [], [], [], [], [], []
shap_values_list, shap_interaction_list = [], []
X_test_list = []
X_test_display_list = []
# cnn = CondensedNearestNeighbour(random_state=42)
# load subject cross validation
features_reader = SubjectCrossValidation()
features_group = "important"
subject_train, subject_test = features_reader.getTrainTestData(1)
# load features
for i in range(len(subject_train)):
    path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\single_episode_features.pkl"

    train_subject = subject_train[i]
    test_subject = subject_test[i]
    features_reader_train = SingleFeaturesReader(path, include_subjects=train_subject)
    features_reader_test = SingleFeaturesReader(path, include_subjects=test_subject)

    X_train, y_train, x_column = features_reader_train.getIndividualObservationData(display=True, features_group=features_group)

    # print('Original dataset shape %s' % Counter(y_train))
    # X_res, y_res = cnn.fit_resample(X_train, y_train)
    # print('Resampled dataset shape %s' % Counter(y_res))

    X_test, y_test, _ = features_reader_test.getIndividualObservationData(display=True, features_group=features_group)

    # clf = BalancedRandomForestClassifier(
    #     sampling_strategy="all", replacement=True, random_state=0, criterion="entropy", max_depth=25, n_estimators=400, bootstrap=False, class_weight="balanced")


    clf = trainXGB(X_train, y_train)

    mcc, cm, acc, auc_score, f1, g_mean = evaluateModel(clf, X_test, y_test)

    mcc_list.append(mcc)
    cm_list.append(cm)
    acc_list.append(acc)
    f1_list.append(f1)
    auc_list.append(auc_score)
    gmean_list.append(g_mean)

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






mcc_all = np.vstack(mcc_list)
cm_all = np.array(cm_list)
acc_all = np.vstack(acc_list)
f1_all = np.array(f1_list)
auc_all = np.array(auc_list)
gmean_all = np.array(gmean_list)

print("%.3f, %.3f, %.3f" % (np.average(acc_all), np.average(mcc_all), np.average(gmean_all)))



# # save shap
X_test_all = pd.concat(X_test_list)
shap_values_all = np.vstack(shap_values_list)
shap_interaction_all = np.vstack(shap_interaction_list)
np.save(single_results_path + "x_gboost_single_shap_values_"+features_group+".npy", shap_values_all)
np.save(single_results_path + "x_gboost_single_shap_interaction_"+features_group+".npy", shap_interaction_all)
X_test_all.to_pickle(single_results_path + "x_gboost_single_X_test_"+features_group+".pkl")
#
#
# # metrics
# np.save(single_results_path + "x_gboost_single_MCC_"+features_group+".npy", mcc_all)
# np.save(single_results_path + "x_gboost_single_confusion_matrix_"+features_group+".npy", cm_all)
# np.save(single_results_path + "x_gboost_single_acc_"+features_group+".npy", acc_all)
# np.save(single_results_path + "x_gboost_single_f1_"+features_group+".npy", f1_all)
# np.save(single_results_path + "x_gboost_single_auc_"+features_group+".npy", auc_all)
# np.save(single_results_path + "x_gboost_single_gmean_"+features_group+".npy", gmean_all)

# shap.summary_plot(shap_values_all, X_test_all, n)
# plt.show()
