import numpy as np
from torch.utils.data import DataLoader
from Double.FeatureReader import SequentialFeaturesReader
from Validation.CrossValidation import DoubleSubjectCrossValidation
import torch
from Models.ActionPerceptionModel import SiamseActionPerceptionModel
from Conf import x_double_features_column
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, recall_score
from imblearn.metrics import geometric_mean_score
from pytorch_metric_learning import losses
from Lib import createDir

torch.manual_seed(3)

path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\double_episode_features.pkl"
scenario = "all_features"
MODEL_PATH = "..\\CheckPoints\\Double_siamse\\" + scenario + "\\"
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

    model = SiamseActionPerceptionModel(input_dim=len(x_double_features_column))
    model.to(device)
    # Set optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.87, 0.999))
    loss_fn = losses.ContrastiveLoss()


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
            loss = loss_fn(outputs.view((-1, 32)), y.view((-1, )))
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
            outputs = model(inputs)
            loss = loss_fn(outputs.view((-1, 32)), y.view((-1,)))

            validation_loss += loss.item()


        return validation_loss / len(val_loader)


    min_val_loss = 1000
    early_stop_idx = 0
    min_gam_last_score = 0
    for j in range(250):

        train_loss = train()
        val_loss = validate()




        print("%.0f - Training loss : %.3f, Validation loss : %.3f" % (j+1, train_loss, val_loss))


    del model
