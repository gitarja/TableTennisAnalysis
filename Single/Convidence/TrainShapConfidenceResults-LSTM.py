import numpy as np
from torch.utils.data import DataLoader
from Single.FeaturesReader import SequentialFeaturesReader
from Validation.CrossValidation import SubjectCrossValidation
import torch
from Models.ActionPerceptionModel import ActionPerceptionModel
from Conf import x_important, x_episode_columns
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, recall_score
from imblearn.metrics import geometric_mean_score
from Models.Losses import FocalLoss, Poly1FocalLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from Lib import createDir

torch.manual_seed(3)

path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\single_episode_features.pkl"
scenario = "ConfidenceInterval"
MODEL_PATH = "..\\..\\CheckPoints\\Single\\" + scenario + "\\"
# check GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

features_reader = SubjectCrossValidation()
subject_train, subject_test = features_reader.getTrainTestData(1)
batch_size = 64
n_window = 3
n_stride = 3
number_of_baggings = 10
index = 0


def customLoss(outputs, y):
    outputs = torch.squeeze(outputs)
    y = torch.squeeze(y)
    loss = loss_fn(outputs, y)
    return loss


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
        outputs = model(inputs)

        loss = customLoss(outputs, y[:, -1])

        gt.append(y.cpu().numpy()[:, -1])
        pred.append(outputs.detach().cpu().numpy() > 0.5)
        probs.append(outputs.detach().cpu().numpy())
        gt_last.append(y.cpu().numpy()[:, -1])
        pred_last.append(outputs.detach().cpu().numpy() > 0.5)
        validation_loss += loss.item()

    mcc_score = matthews_corrcoef(np.concatenate(gt_last).flatten(), np.concatenate(pred_last).flatten())
    acc_score = balanced_accuracy_score(np.concatenate(gt_last).flatten(), np.concatenate(pred_last).flatten())
    gam_score = geometric_mean_score(np.concatenate(gt).flatten(), np.concatenate(pred).flatten(),
                                     average="binary")

    return validation_loss / len(val_loader), acc_score, mcc_score, gam_score
def train():
    training_loss = 0.

    model.train()
    for i_batch, sample_batched in enumerate(train_loader):
        inputs = sample_batched["inputs"].type(torch.float).to(device)
        y = sample_batched["label"].to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = customLoss(outputs, y[:, -1])
        loss.backward()

        # Adjust learning weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        training_loss += loss.item()

    return training_loss / len(train_loader)

for i in range(len(subject_train)):

    val_dataset = SequentialFeaturesReader(path, include_subjects=subject_test[i], n_window=n_window, n_stride=n_stride)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    for j in range(number_of_baggings+1):
        CHECK_POINT_BEST_PATH = MODEL_PATH + "model_best_" + str(index) + ".pkl"
        CHECK_POINT_PATH = MODEL_PATH + "model_" + str(index) + ".pkl"


        acc_all = []
        mcc_all = []
        gam_all = []
        gam_last_all = []
        train_loss_all = []
        val_loss_all = []
        if j >0:
            bagging = True
        else:
            bagging = False
        train_dataset = SequentialFeaturesReader(path, include_subjects=subject_train[i], n_window=n_window, n_stride=n_stride, training=False, bagging=bagging)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=0)



        model = ActionPerceptionModel(input_dim=len(x_important), sequence_len=n_window)
        model.to(device)
        # Set optimizer and loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0085, betas=(0.9, 0.999), weight_decay=1e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=1, eta_min=1e-4)
        loss_fn = Poly1FocalLoss(gamma=2., alpha=.25, num_classes=2, reduction='mean', label_is_onehot=True)

        min_val_loss = 1000
        early_stop_idx = 0
        acc_best = 0
        mcc_best = 0
        gam_best = 0
        model_best = None
        for k in range(100):

            train_loss = train()
            scheduler.step()
            val_loss, acc_score, mcc_score, gam_score = validate()


            if (mcc_score >= mcc_best) :
                min_val_loss = val_loss
                early_stop_idx = 0

                acc_best = acc_score
                mcc_best = mcc_score
                gam_best = gam_score
                torch.save({
                    'epoch': k,
                    'model_state_dict': model.state_dict(),
                }, CHECK_POINT_BEST_PATH)

            else:
                early_stop_idx += 1
                torch.save({
                    'epoch': j,
                    'model_state_dict': model.state_dict(),
                }, CHECK_POINT_PATH)
        print(CHECK_POINT_BEST_PATH)
        print("%.3f, %.3f, %.3f" % (acc_best, mcc_best, gam_best))
        del model
        torch.cuda.empty_cache()
        index += 1
