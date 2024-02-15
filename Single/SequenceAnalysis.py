import pandas as pd
from pomegranate.distributions import Uniform, Normal, Categorical
from pomegranate.hmm import DenseHMM, SparseHMM
from pomegranate.bayesian_network import BayesianNetwork

import numpy as np
from Conf import x_episode_columns
from FeaturesReader import SingleFeaturesReader
import matplotlib.pyplot as plt
#case we have a sequence of ACGT

# load features
path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\single_episode_features.pkl"
experiment_data_path = "/\\Experiment\\"
train_list = pd.read_csv(experiment_data_path + "single_train_0.csv")
test_list = pd.read_csv(experiment_data_path + "single_test_0.csv")

train_subject = train_list.loc[:, "Subject1"].values
test_subject = test_list.loc[:, "Subject1"].values
features_reader_train = SingleFeaturesReader(path, include_subjects=train_subject)
features_reader_test = SingleFeaturesReader(path, include_subjects=test_subject)

X_train, y_train = features_reader_train.getAllData(train=True)

X_test, y_test = features_reader_test.getAllData(train=False)

n_features = len(x_episode_columns)

d1 = Normal(covariance_type="diag") # uniform for all character
d2 = Normal(covariance_type="diag") # prefer CG over AT


model = DenseHMM([d1, d2], verbose=True, max_iter=200)
model.add_edge(d1, model.end, 0.01)
model.add_edge(d2, model.end, 0.01)
model.fit(X=X_train, priors=y_train)

print(model.ends)
idx_test = 0
X_test_sample = np.concatenate([X_test[idx_test]])
y_test_sample = np.argmax(np.concatenate([ y_test[idx_test]]), axis=1)
plt.plot(model.predict_proba(np.expand_dims(X_test_sample, 0))[0], label=['Fail', 'Success'])
plt.plot(y_test_sample, label='GT')
plt.legend()
plt.show()