import numpy as np
from torch.utils.data import DataLoader
from Double.FeatureReader import SequentialFeaturesReader
from Validation.CrossValidation import DoubleSubjectCrossValidation
import torch
from Models.ActionPerceptionModel import ActionPerceptionModel
from Conf import x_double_features_column
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, recall_score
from imblearn.metrics import geometric_mean_score
from Models.Losses import FocalLoss
from Lib import createDir

torch.manual_seed(3)

path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\double_episode_features.pkl"
scenario = "all_features"
MODEL_PATH = "..\\CheckPoints\\Double\\" + scenario + "\\"
# check GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

features_reader = DoubleSubjectCrossValidation()
features_group = "perception-important"
subject_train, subject_test = features_reader.getTrainTestData(1)
batch_size = 32
n_window = 10
createDir(MODEL_PATH)
for i in range(len(subject_train)):
    CHECK_POINT_BEST_PATH = MODEL_PATH + "model_best_"+str(i)+".pkl"
    CHECK_POINT_PATH = MODEL_PATH + "model_" + str(i) + ".pkl"


    acc_all = []
    mcc_all = []
    gam_all = []
    gam_last_all = []
    train_dataset = SequentialFeaturesReader(path, include_subjects=subject_train[i], n_window=n_window, n_stride=n_window)
    val_dataset = SequentialFeaturesReader(path, include_subjects=subject_test[i], n_window=n_window, n_stride=n_window)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    model = ActionPerceptionModel(input_dim=len(x_double_features_column))
    model.to(device)
    # Set optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.87, 0.999))
    post_weight = torch.Tensor([25.5, 1.]).to(device)
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = FocalLoss(gamma=2, weights=post_weight)


    def train():
        training_loss = 0.
        model.train()
        for i_batch, sample_batched in enumerate(train_loader):
            inputs = sample_batched["inputs"].type(torch.float).to(device)
            y = sample_batched["label"].to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch

            outputs = torch.sigmoid(model(inputs))

            # Compute the loss and its gradients
            loss = loss_fn(torch.flatten(outputs), torch.flatten(y))
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            training_loss += loss.item()

        return training_loss / len(train_loader)


    def validate(print=False):
        validation_loss = 0
        model.eval()
        gt = []
        pred = []
        probs = []
        gt_last = []
        pred_last = []
        for i_batch, sample_batched in enumerate(val_loader):
            inputs = sample_batched["inputs"].type(torch.float).to(device)
            y = sample_batched["label"].to(device)

            # Make predictions for this batch
            outputs = torch.sigmoid(model(inputs))



            loss = loss_fn(torch.flatten(outputs), torch.flatten(y))

            gt.append(y.cpu().numpy())
            pred.append(outputs.detach().cpu().numpy() > 0.5)
            probs.append(outputs.detach().cpu().numpy())
            gt_last.append(y.cpu().numpy()[:,-1, :])
            pred_last.append(outputs.detach().cpu().numpy()[:,-1, :] > 0.5)
            validation_loss += loss.item()
        mcc_score = matthews_corrcoef(np.asarray(gt).flatten(), np.asarray(pred).flatten())
        acc_score = balanced_accuracy_score(np.asarray(gt).flatten(), np.asarray(pred).flatten())
        gam_score = geometric_mean_score(np.asarray(gt).flatten(), np.asarray(pred).flatten(), average="binary")
        gam_score_last  = geometric_mean_score(np.asarray(gt_last).flatten(), np.asarray(pred_last).flatten(), average="binary")
        pred = np.vstack(pred).squeeze()
        gt = np.vstack(gt).squeeze()
        probs = np.vstack(probs).squeeze()
        return validation_loss / len(val_loader), acc_score, mcc_score, gam_score, gam_score_last


    min_val_loss = 1000
    early_stop_idx = 0
    min_gam_last_score = 0
    for j in range(250):

        train_loss = train()
        val_loss, acc_score, mcc_score, gam_score, gam_last_score = validate()
        acc_all.append(acc_score)
        mcc_all.append(mcc_score)
        gam_all.append(gam_score)
        gam_last_all.append(gam_last_score)
        if (val_loss <= min_val_loss) & (gam_last_score >= min_gam_last_score):
            min_val_loss = val_loss
            min_gam_last_score = gam_last_score
            early_stop_idx = 0

            torch.save({
                'epoch': j,
                'model_state_dict': model.state_dict(),
            }, CHECK_POINT_BEST_PATH)

        else:
            early_stop_idx += 1
            torch.save({
                'epoch': j,
                'model_state_dict': model.state_dict(),
            }, CHECK_POINT_PATH)


        # print("%.0f - Training loss : %.3f, Validation loss : %.3f, ACC : %.3f MCC : %.3f, GAM: %.3f, GAM_LAST: %.3f" % (j+1, train_loss, val_loss, acc_score, mcc_score, gam_score, gam_last_score))

    print("%.3f, %.3f, %3.f, %.3f" % (np.max(acc_all), np.max(mcc_all), np.max(gam_all), np.max(gam_last_all)))
    del model
