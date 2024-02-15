import numpy as np
from torch.utils.data import DataLoader
from Single.FeaturesReader import SequentialFeaturesReader
from Validation.CrossValidation import SubjectCrossValidation
import torch
from Models.ActionPerceptionModel import SiamseActionPerceptionModel
from Conf import x_episode_columns, x_important
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, recall_score
from imblearn.metrics import geometric_mean_score
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity

from Lib import createDir

torch.manual_seed(3)

path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\single_episode_features.pkl"
scenario = "all_features"
MODEL_PATH = "..\\CheckPoints\\Single_siamse\\" + scenario + "\\"
# check GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

features_reader = SubjectCrossValidation()
features_group = "perception-important"
subject_train, subject_test = features_reader.getTrainTestData(1)
batch_size = 32
n_window = 3
n_stride = 1
emb_dim = 128
createDir(MODEL_PATH)
for i in range(len(subject_train)):
    CHECK_POINT_BEST_PATH = MODEL_PATH + "model_best_" + str(i) + ".pkl"
    CHECK_POINT_PATH = MODEL_PATH + "model_" + str(i) + ".pkl"

    acc_all = []
    mcc_all = []
    gam_all = []
    train_loss_all = []
    val_loss_all = []
    train_dataset = SequentialFeaturesReader(path, include_subjects=subject_train[i], n_window=n_window,
                                             n_stride=n_stride, training=False)
    val_dataset = SequentialFeaturesReader(path, include_subjects=subject_test[i], n_window=n_window, n_stride=n_stride)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    model = SiamseActionPerceptionModel(input_dim=len(x_important))
    model.to(device)
    # Set optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = losses.TripletMarginLoss(margin=1.5, swap=False, distance=CosineSimilarity(), smooth_loss=True)
    miner = miners.MultiSimilarityMiner()


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
            hard_pairs = miner(outputs.view((-1, emb_dim)), y.view((-1,)))

            # Compute the loss and its gradients
            loss = loss_fn(outputs.view((-1, emb_dim)), y.view((-1,)), hard_pairs)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            training_loss += loss.item()

        return training_loss / len(train_loader)


    def validate(print=False):
        validation_loss = 0
        model.eval()
        y_train_all = []
        embds_all = []
        for i_batch, sample_batched in enumerate(train_loader):
            X_train = sample_batched["inputs"].type(torch.float).to(device)
            y = sample_batched["label"].to(device)
            embds = model(X_train)
            y_train_all.append(y)
            embds_all.append(embds)

        y_train_all = torch.concatenate(y_train_all, dim=0).view((-1,))
        embds_all = torch.concatenate(embds_all, dim=0).view((-1, emb_dim))

        model.setEmbedding(embds_all, y_train_all)

        gt_last = []
        pred_last = []
        for i_batch, sample_batched in enumerate(val_loader):
            inputs = sample_batched["inputs"].type(torch.float).to(device)
            y = sample_batched["label"].to(device)

            # Make predictions for this batch
            outputs = model(inputs)
            preds = model.predict(inputs, n=10)
            loss = loss_fn(outputs.view((-1, emb_dim)), y.view((-1,)))

            gt_last.append(y.cpu().numpy()[:, -1])
            pred_last.append(preds.detach().cpu().numpy()[:, -1])

            validation_loss += loss.item()

        mcc_score = matthews_corrcoef(np.concatenate(gt_last).flatten(), np.concatenate(pred_last).flatten())
        acc_score = balanced_accuracy_score(np.concatenate(gt_last).flatten(), np.concatenate(pred_last).flatten())
        gam_score = geometric_mean_score(np.concatenate(gt_last).flatten(), np.concatenate(pred_last).flatten(),
                                         average="binary")

        return validation_loss / len(val_loader), acc_score, mcc_score, gam_score


    min_val_loss = 1000
    early_stop_idx = 0
    min_gam_last_score = 0
    for j in range(100):

        train_loss = train()
        val_loss, acc_score, mcc_score, gam_score = validate()

        train_loss_all.append(train_loss)
        val_loss_all.append(val_loss)

        if (val_loss <= min_val_loss):
            min_val_loss = val_loss
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

        print(
            "%.0f - Training loss : %.3f, Validation loss : %.3f, ACC : %.3f MCC : %.3f, GAM: %.3f" % (
                j + 1, train_loss, val_loss, acc_score, mcc_score, gam_score))

    import matplotlib.pyplot as plt

    plt.plot(train_loss_all, label="train")
    plt.plot(val_loss_all, label="val")
    plt.legend()
    plt.show()
    del model
