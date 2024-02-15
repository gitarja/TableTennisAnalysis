import pandas as pd
import numpy as np
from FeaturesReader import SingleFeaturesReader
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, f1_score, roc_curve, auc
from imblearn.under_sampling import CondensedNearestNeighbour
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import shap
import matplotlib.pyplot as plt
from Validation.CrossValidation import SubjectCrossValidation
from Conf import single_results_path
from sklearn.model_selection import train_test_split
import pickle
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os
from imblearn.metrics import geometric_mean_score
from sklearn.ensemble import BaggingClassifier

os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'

#case we have a sequence of ACGT
np.random.seed(1945)

# prepare model for training

def trainModel(X, y):
    model = LogisticRegression(random_state=0, class_weight="balanced", max_iter=10000, penalty="l1", solver="liblinear").fit(X, y)
    return model


def trainModelBaggin(x, y):

    model = RandomForestClassifier(class_weight="balanced", max_depth=10, n_estimators=50, max_samples=.5, max_features=1., max_leaf_nodes=15).fit(x, y)
    return model

def evaluateModel(model, X_test, y_test):

    y_pred = model.predict_proba(X_test)[:, 1]
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
# load subject cross validation
features_reader = SubjectCrossValidation()
model_name = "bagging"
features_group = "all"
subject_train, subject_test = features_reader.getTrainTestData(1)
# load features
for i in range(len(subject_train)):
    path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\single_episode_features.pkl"

    train_subject = subject_train[i]
    test_subject = subject_test[i]
    features_reader_train = SingleFeaturesReader(path, include_subjects=train_subject)
    features_reader_test = SingleFeaturesReader(path, include_subjects=test_subject)

    X_train, y_train, x_column = features_reader_train.getIndividualObservationData(display=False, features_group=features_group)


    X_test, y_test, _ = features_reader_test.getIndividualObservationData(display=False, features_group=features_group)

    X_test_display, _, _ = features_reader_test.getIndividualObservationData(display=True, features_group=features_group)

    clf = trainModelBaggin(X_train, y_train)

    mcc, cm, acc, auc_score, f1, g_mean = evaluateModel(clf, X_test, y_test)

    mcc_list.append(mcc)
    cm_list.append(cm)
    acc_list.append(acc)
    f1_list.append(f1)
    auc_list.append(auc_score)
    gmean_list.append(g_mean)

    # # compute shap
    X_train_summary = shap.kmeans(X_train, 20)
    explainer = shap.KernelExplainer(clf.predict, X_train_summary)
    shap_values = explainer.shap_values(X_test)

    # append test data
    X_test_list.append(X_test_display)
    # append results
    shap_values_list.append(shap_values)








mcc_all = np.vstack(mcc_list)
cm_all = np.array(cm_list)
acc_all = np.vstack(acc_list)
f1_all = np.array(f1_list)
auc_all = np.array(auc_list)
gmean_all = np.array(gmean_list)

print("%.3f, %.3f, %.3f" % (np.average(acc_all), np.average(mcc_all), np.average(gmean_all)))



# # # # save shap
X_test_all = pd.concat(X_test_list)
shap_values_all = np.vstack(shap_values_list)
np.save(single_results_path + "x_"+model_name+"_single_shap_values_"+features_group+".npy", shap_values_all)
X_test_all.to_pickle(single_results_path + "x_"+model_name+"_single_X_test_"+features_group+".pkl")


# metrics
np.save(single_results_path + "x_"+model_name+"_single_MCC_"+features_group+".npy", mcc_all)
np.save(single_results_path + "x_"+model_name+"_single_confusion_matrix_"+features_group+".npy", cm_all)
np.save(single_results_path + "x_"+model_name+"_single_acc_"+features_group+".npy", acc_all)
np.save(single_results_path + "x_"+model_name+"_single_f1_"+features_group+".npy", f1_all)
np.save(single_results_path + "x_"+model_name+"_single_auc_"+features_group+".npy", auc_all)
np.save(single_results_path + "x_"+model_name+"_single_gmean_"+features_group+".npy", gmean_all)

