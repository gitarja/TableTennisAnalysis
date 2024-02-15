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

torch.backends.cudnn.enabled=False

torch.manual_seed(3)

path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\single_episode_features.pkl"
scenario = "all-features"
MODEL_PATH = "..\\CheckPoints\\Single\\" + scenario + "\\"
# check GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

features_reader = SubjectCrossValidation()
subject_train, subject_test = features_reader.getTrainTestData(2)
batch_size = 256
n_window = 3
n_stride = 3
createDir(MODEL_PATH)
X_test_list_disp = []
shap_values_list = []
labels = np.expand_dims(np.array([str(i) for i in range(n_window)]), -1)
model_name = "LSTM"
features_group = "all"
features = x_important
for i in range(len(subject_train)):
    CHECK_POINT_BEST_PATH = MODEL_PATH + "model_best_"+str(i)+".pkl"
    CHECK_POINT_PATH = MODEL_PATH + "model_" + str(i) + ".pkl"

    acc_all = []
    mcc_all = []
    gam_all = []
    gam_last_all = []
    train_loss_all = []
    val_loss_all = []
    train_dataset = SequentialFeaturesReader(path, include_subjects=subject_train[i], n_window=n_window, n_stride=n_stride, training=False)
    val_dataset = SequentialFeaturesReader(path, include_subjects=subject_test[i], n_window=n_window, n_stride=n_stride, training=False)
    val_dataset_display = SequentialFeaturesReader(path, include_subjects=subject_test[i], n_window=n_window, n_stride=n_stride, display=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    val_loader_display = DataLoader(val_dataset_display, batch_size=1,
                            shuffle=False, num_workers=0)

    model = ActionPerceptionModel(input_dim=len(features), inference=True, sequence_len=n_window)
    checkpoint = torch.load(CHECK_POINT_BEST_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    torch.set_grad_enabled(True)



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
    for i_batch, (sample_batched, sample_batched_disp) in enumerate(zip(val_loader, val_loader_display)):
        X_test = sample_batched["inputs"].type(torch.float).to(device)
        X_test_all.append(X_test)
        y = sample_batched["label"][:, -1]
        y_test_all.append(y)

        X_test_all_disp = sample_batched_disp["inputs"].numpy().astype(float)
        data = np.hstack([np.reshape(X_test_all_disp, (n_window, len(features))), labels])
        df_append = pd.DataFrame(data, columns=features + ["n_pos"])
        X_test_list_disp.append(df_append)

    X_test_all = torch.concatenate(X_test_all, dim=0)
    y_test_all = np.concatenate(y_test_all)

    X = torch.concatenate([X_test_all, X_train_all])
    # compute shap

    e = GradientShap(model)

    shap_values = e.attribute(X_test_all, baselines=torch.mean(X_train_all, dim=0, keepdim=True))
    shap_values_list.append(shap_values.detach().cpu().numpy())

    # # append results
    # for i in range(1+ (len(X_test_all) // 100)):
    #    shap_values = e.attribute(X_test_all[(i*100):(i+1) * 100], X_test_all[(i*100):(i+1) * 100] * 0)
    #    shap_values_list.append(shap_values.detach().cpu().numpy())

    del model, checkpoint
    # gc.collect()
    # torch.cuda.empty_cache()


X_test_all_disp = pd.concat(X_test_list_disp)
shap_values_all = np.vstack(shap_values_list)

np.save(single_results_path + "x_"+model_name+"_single_shap_values_"+features_group+".npy", shap_values_all)
X_test_all_disp.to_pickle(single_results_path + "x_"+model_name+"_single_X_test_"+features_group+".pkl")