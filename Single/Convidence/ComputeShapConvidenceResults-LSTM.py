import numpy as np
from torch.utils.data import DataLoader
from Single.FeaturesReader import SequentialFeaturesReader
from Validation.CrossValidation import SubjectCrossValidation
import torch
from Models.ActionPerceptionModel import ActionPerceptionModel
from Conf import x_important, x_episode_columns

from Lib import createDir
# import shap
from captum.attr import GradientShap, DeepLiftShap, ShapleyValueSampling
import pandas as pd
from Conf import single_results_path
import gc

torch.backends.cudnn.enabled = False

torch.manual_seed(3)


def correlationCoeff(arr1, arr2):
    # Standardize the arrays
    arr1_standardized = (arr1 - np.mean(arr1)) / np.std(arr1)
    arr2_standardized = (arr2 - np.mean(arr2)) / np.std(arr2)

    # Compute the dot product
    dot_product = np.dot(arr1_standardized, arr2_standardized)

    # Calculate the correlation coefficient
    correlation = dot_product / len(arr1)

    return correlation


path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\single_episode_features.pkl"
scenario = "ConfidenceInterval"
MODEL_PATH = "..\\..\\CheckPoints\\Single\\" + scenario + "\\"
# check GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

features_reader = SubjectCrossValidation()
subject_train, subject_test = features_reader.getTrainTestData(2)
batch_size = 256
n_window = 3
n_stride = 3
number_of_baggings = 10
index = 0
index_results = 0
X_test_list_disp = []

labels = np.expand_dims(np.array([str(i) for i in range(n_window)]), -1)
model_name = "LSTM"
features_group = "all"
features = x_important
bootstrap_results = np.zeros((number_of_baggings * 5, len(x_important)))
for i in range(len(subject_train)):


    train_dataset = SequentialFeaturesReader(path, include_subjects=subject_train[i], n_window=n_window,
                                             n_stride=n_stride, training=False, bagging=False)

    val_dataset = SequentialFeaturesReader(path, include_subjects=subject_test[i], n_window=n_window, n_stride=n_stride,
                                           training=False)
    val_dataset_display = SequentialFeaturesReader(path, include_subjects=subject_test[i], n_window=n_window,
                                                   n_stride=n_stride, display=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    # load background
    X_train_all = []

    for i_batch, sample_batched in enumerate(train_loader):
        X_train = sample_batched["inputs"].type(torch.float).to(device)
        X_train_all.append(X_train)
    X_train_all = torch.concatenate(X_train_all, dim=0)

    # load test
    X_test_all = []
    y_test_all = []
    X_test_all_disp = []
    for i_batch, sample_batched in enumerate(val_loader):
        X_test = sample_batched["inputs"].type(torch.float).to(device)
        X_test_all.append(X_test)
        y = sample_batched["label"][:, -1]
        y_test_all.append(y)

    X_test_all = torch.concatenate(X_test_all, dim=0)
    y_test_all = np.concatenate(y_test_all)
    shap_values_list = []
    for j in range(number_of_baggings + 1):
        CHECK_POINT_BEST_PATH = MODEL_PATH + "model_best_" + str(index) + ".pkl"
        CHECK_POINT_PATH = MODEL_PATH + "model_" + str(index) + ".pkl"
        print(CHECK_POINT_BEST_PATH)

        model = ActionPerceptionModel(input_dim=len(features), inference=True, sequence_len=n_window)
        checkpoint = torch.load(CHECK_POINT_BEST_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        torch.set_grad_enabled(True)

        # compute shap
        e = GradientShap(model)

        shap_values = e.attribute(X_test_all, baselines=torch.mean(X_train_all, dim=0, keepdim=True))
        shap_values_list.append(shap_values.detach().cpu().numpy())

        del model, checkpoint

        index += 1

    for k in range(1, number_of_baggings + 1):
        baseline = shap_values_list[0]
        comparison = shap_values_list[k]
        for l in range(len(x_important)):
            corr = correlationCoeff(baseline[:, 2, l], comparison[:, 2, l])
            bootstrap_results[index_results, l] = corr
        index_results += 1

np.savetxt(single_results_path + "confidence_shap_LSTM_2_" + features_group + ".csv", bootstrap_results, delimiter=",")
