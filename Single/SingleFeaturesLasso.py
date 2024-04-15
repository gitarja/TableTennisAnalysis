import pandas as pd
import numpy as np

from Conf import x_episode_columns
from FeaturesReader import SingleFeaturesReader
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, f1_score, roc_curve, auc
from imblearn.under_sampling import CondensedNearestNeighbour
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
import matplotlib.pyplot as plt
from Validation.CrossValidation import SubjectCrossValidation
from Conf import single_results_path
from Models.LogisticRegSum import LogisticRegression
import statsmodels.api as sm
import shap
import pickle
from imblearn.metrics import geometric_mean_score
from sklearn.ensemble import RandomForestClassifier

# case we have a sequence of ACGT


# prepare model for training

def trainLASSO(X_train, y_train, alpha=0.1):
    model = LogisticRegression(penalty='elasticnet', solver="saga", class_weight="balanced", fit_intercept=True,
                               max_iter=100000, l1_ratio=alpha)

    model.fit(X_train.values, y_train)

    return model


def trainModelBaggin(x, y, depth):

    model = RandomForestClassifier(class_weight="balanced", max_depth=depth, n_estimators=100, max_samples=.5, max_features=1., max_leaf_nodes=15)
    model.fit(x.values, y)
    return model


def evaluateModel(model, X_test, y_test, x_column):
    y_pred = model.predict(X_test.values)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    acc = balanced_accuracy_score(y_test, y_pred)
    # fpr, tpr, thresholds = roc_curve(y_test, model.model.predict_proba(X_test.values)[:, 1], pos_label=1)
    g_mean = geometric_mean_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred, average='weighted')


    return mcc, cm, acc, g_mean, f1


features_reader = SubjectCrossValidation()
path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\single_episode_features.pkl"

subject_train, subject_test = features_reader.getTrainTestData(1)

for features_group in ["all"]:

    all_acc, all_mcc, all_gmean = [], [], []
    # load subject cross validation
    mcc_list, cm_list, acc_list, gmean_list, f1_list, statistic_list = [], [], [], [], [], []

    X_test_list = []
    X_test_display_list = []
    for a in [ 5, 10, 15, 20, 25]:
        # load features
        for i in range(len(subject_train)):
            train_subject = subject_train[i]
            test_subject = subject_test[i]
            features_reader_train = SingleFeaturesReader(path, include_subjects=train_subject)
            features_reader_test = SingleFeaturesReader(path, include_subjects=test_subject)

            X_train, y_train, x_column = features_reader_train.getIndividualObservationData(display=False,
                                                                                            features_group=features_group)

            X_test, y_test, _ = features_reader_test.getIndividualObservationData(display=False,
                                                                                  features_group=features_group)
            X_test_disp, y_test_disp, _ = features_reader_test.getIndividualObservationData(display=True,
                                                                                            features_group=features_group)

            clf = trainModelBaggin(X_train, y_train, depth=a)
            mcc, cm, acc, gmean_score, f1 = evaluateModel(clf, X_test, y_test, x_column)


            # append results

            acc_list.append(acc)
            mcc_list.append(mcc)
            cm_list.append(cm)
            gmean_list.append(gmean_score)
            f1_list.append(f1)


        mcc_all = np.vstack(mcc_list)
        acc_all = np.vstack(acc_list)
        cm_all = np.array(cm_list)
        gmean_all = np.vstack(gmean_list)
        f1_all = np.array(f1_list)

        print("alpha", a)


        all_acc.append(np.average(acc_all))
        all_mcc.append(np.average(mcc_all))
        all_gmean.append(np.average(gmean_all))
        print("---------------------------")

    print(all_acc)
    print(all_mcc)
    print(all_gmean)
    # print(cm_all)
    # save the evaluation metrices
    # np.save(single_results_path + "single_MCC_" + features_group + ".npy", mcc_all)
    # np.save(single_results_path + "single_ACC_" + features_group + ".npy", acc_all)
    # np.save(single_results_path + "single_confusion_matrix_" + features_group + ".npy", cm_all)
    # np.save(single_results_path + "auc_score_" + features_group + ".npy", auc_all)
    # np.save(single_results_path + "f1_score_" + features_group + ".npy", f1_all)
    #
    # # save statistic files
    # with open(single_results_path + "single_statistic_" + features_group + ".pkl", 'wb') as f:
    #     pickle.dump(statistic_list, f, protocol=4)


# shap_values_all = np.vstack(shap_values_list)
# X_test_all = pd.concat(X_test_list)
# mcc_all = np.vstack(mcc_list)
# acc_all = np.vstack(acc_list)
# cm_all = np.array(cm_list)
#
# print(np.average(mcc_all))
# print(np.average(acc_all))
# # save shap
# np.save(single_results_path + "single_shap_values.npy", shap_values_all)
# X_test_all.to_pickle(single_results_path + "single_X_test.pkl")
# np.save(single_results_path + "single_MCC.npy", mcc_all)
# np.save(single_results_path + "single_ACC.npy", acc_all)
# np.save(single_results_path + "single_confusion_matrix.npy", cm_all)
#
# # save statistic files
# with open(single_results_path +'single_statistic.pkl', 'wb') as f:
#     pickle.dump(statistic_list, f, protocol=4)
#
