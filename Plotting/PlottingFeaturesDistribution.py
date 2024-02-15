import pandas as pd
import numpy as np
from Single.FeaturesReader import SingleFeaturesReader
from Validation.CrossValidation import SubjectCrossValidation
import seaborn as sns
import matplotlib.pyplot as plt

# case we have a sequence of ACGT
np.random.seed(1945)

features_reader = SubjectCrossValidation()

subject_train, subject_test = features_reader.getTrainTestData(1)
# load features

path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\single_episode_features.pkl"

train_subject = subject_train[0]
test_subject = subject_test[0]
subjects = np.concatenate([train_subject,  test_subject])
features_reader = SingleFeaturesReader(path, include_subjects=subjects)

X, y, x_column = features_reader.getIndividualObservationData(display=True, features_group="important", label=True)


fig, axs = plt.subplots(4, 4, figsize=(7, 7))

sns.histplot(data=X, x="im_rack_wrist_dist", kde=False, color="skyblue", ax=axs[0, 0])
sns.histplot(data=X, x="im_racket_force", kde=False, color="skyblue", ax=axs[0, 1])
sns.histplot(data=X, x="im_rb_ang_collision", kde=False, color="skyblue", ax=axs[0, 2])
sns.histplot(data=X, x="im_rb_dist", kde=False, color="skyblue", ax=axs[0, 3])


sns.histplot(data=X, x="ec_fs_ball_racket_dir", kde=False, color="skyblue", ax=axs[1, 0])
sns.histplot(data=X, x="ec_start_fs", kde=False, color="skyblue", ax=axs[1, 1])
sns.histplot(data=X, x="pr_p1_al_on", kde=False, color="skyblue", ax=axs[1, 2])
sns.histplot(data=X, x="pr_p1_al_miDo", kde=False, color="skyblue", ax=axs[1, 3])


sns.histplot(data=X, x="pr_p1_al_gM", kde=False, color="skyblue", ax=axs[2, 0])
sns.histplot(data=X, x="pr_p2_al_on", kde=False, color="skyblue", ax=axs[2, 1])
sns.histplot(data=X, x="pr_p2_al_miDo", kde=False, color="skyblue", ax=axs[2, 2])
sns.histplot(data=X, x="pr_p2_al_gM", kde=False, color="skyblue", ax=axs[2, 3])


sns.histplot(data=X, x="pr_p3_fx_on", kde=False, color="skyblue", ax=axs[3, 0])
plt.tight_layout()
plt.show()





# sampled = X.groupby('success')
# X_balance = sampled.apply(lambda x: x.sample(sampled.size().min()).reset_index(drop=True))
#
# g = sns.FacetGrid(X_balance)
# g.map_diag(sns.histplot)
# g.map_offdiag(sns.scatterplot)
# plt.show()