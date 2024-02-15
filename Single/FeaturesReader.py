import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from Conf import x_episode_columns, y_episode_column, normalize_x_episode_columns, x_important, x_perception, y_regression_column
import numpy as np
import torch
import random
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from Conf import single_summary_path
from torch.utils.data import Dataset
np.random.seed(1945)


def toCategorical(y, num_classes=2):
    """ 1-hot encodes a tensor """
    y = y.astype(int)
    return np.eye(num_classes, dtype='uint8')[y]


def converToDiscreet(df):
    # df = df.fillna(-1945)
    trans_norm = KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='quantile', subsample=None)
    for column in x_important:
        # fill in the non nan
        df.loc[~ np.isnan(df.loc[:, column]), column] = trans_norm.fit_transform(
            np.expand_dims(df.loc[~ np.isnan(df.loc[:, column]), column].values, -1)).flatten()
        # fill in the nan
        df.loc[np.isnan(df.loc[:, column]), column] = len(
            np.unique(df.loc[~ np.isnan(df.loc[:, column]), column].values))

    return df


class SingleFeaturesReader:

    def __init__(self, file_path="", include_subjects=["test"], n_window=3, discretization=False):
        df = pd.read_pickle(file_path)

        if discretization:
            df = converToDiscreet(df)

        self.mean = np.nanmean(
            df.loc[:, normalize_x_episode_columns], axis=0)
        self.std = np.nanstd(
            df.loc[:, normalize_x_episode_columns], axis=0)
        # select subjects subjects
        df = df.loc[df["id_subject"].isin(include_subjects), :]
        # exclude -1 data
        df = df.loc[df["success"] != -1]
        self.n_window = n_window
        self.df = df

    def splitEpisode(self, v):
        min_seq = self.n_window
        X_sequence = []

        x_columns = x_episode_columns + y_episode_column
        # x_columns = x_episode_columns + y_regression_column
        if len(v) > min_seq:
            # for i in range(0, (len(v) - min_seq)+1):
            #     features = np.concatenate(v.iloc[i:(i + min_seq)][x_columns].values)
            #
            #     X_sequence.append(features)

            for i in range(0, 1):
                features = np.concatenate(v.iloc[-min_seq:][x_columns].values)

                X_sequence.append(features)
            colnames = []
            for t in range(min_seq):
                colnames.extend([(x, t) for x in x_columns])
            df = pd.DataFrame(np.asarray(X_sequence), columns=colnames)
            return df
        else:
            return None

    def constructEpisodes(self, df, train=False):
        subjects_group = df.groupby(['id_subject'])
        X_all = []
        for s in subjects_group:
            for e in s[1].groupby(['episode_label']):
                X_seq = self.splitEpisode(e[1])
                if X_seq is not None:
                    X_all.append(X_seq)

        X_all = pd.concat(X_all, ignore_index=True)
        return X_all

    def contructMixEpisode(self, df):

        subjects_group = df.groupby(['id_subject'])
        X_all = []
        y_all = []
        for s in subjects_group:
            X_seq, y_seq = self.splitEpisode(s[1], th=50, augment=20, min_seq=2)
            X_all = X_all + X_seq
            y_all = y_all + y_seq

        return X_all, y_all

    def getAllData(self, train=False):
        X1 = self.constructEpisodes(self.df, train)
        return X1

    def normalizeDF(self, df, display=False):
        df = df.copy()

        if display == False:
            df.loc[:, normalize_x_episode_columns] = (df.loc[:, normalize_x_episode_columns] - self.mean) / self.std

            df = df.fillna(0)
        return df

    def getIndividualObservationData(self, display=False, features_group="all", label=False):
        '''
        :param display:
        :param features_group:
        all : all combination
        per_ec : perception + execution
        per_im : perception + impact
        per : perception
        :return:
        '''
        df = self.normalizeDF(self.df, display)
        x_column = x_episode_columns
        if features_group == "important":
            x_column = x_important
        # x_column = x_episode_columns


        if display == False:
            df.loc[:, normalize_x_episode_columns] = (df.loc[:, normalize_x_episode_columns] - self.mean) / self.std

            df = df.fillna(df.mean())

        if label:
            x_column = x_column + y_episode_column

        X = df.iloc[:][x_column]
        y = df.iloc[:][y_episode_column].values.ravel()

        return X, y, x_column


class SequentialFeaturesReader(Dataset):

    def __init__(self, file_path="", include_subjects=["test"], n_window=3, n_stride = 5, training=False, display=False):
        df = pd.read_pickle(file_path)
        self.columns = x_important
        self.mean = np.nanmean(
            df.loc[:, normalize_x_episode_columns], axis=0)
        self.std = np.nanstd(
            df.loc[:, normalize_x_episode_columns], axis=0)
        if display == False:
            df = self.normalizeDF(df)
        # select subjects subjects
        df = df.loc[df["id_subject"].isin(include_subjects), :]
        # exclude -1 data
        df = df.loc[df["success"] != -1]
        self.n_window = n_window
        self.n_stride = n_stride
        self.df = df
        self.training = training

        self.X, self.y = self.getAllData()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'inputs': self.X[idx], 'label': self.y[idx]}

        return sample

    def normalizeDF(self, df):
        df = df.copy()

        df.loc[:, normalize_x_episode_columns] = (df.loc[:, normalize_x_episode_columns] - self.mean) / self.std
        df = df.fillna(df.mean())

        return df

    def getAllData(self):
        X, y = self.constructEpisodes(self.df)
        return X, y

    def constructEpisodes(self, df):
        subjects_group = df.groupby(['id_subject'])
        X_all = []
        y_all = []
        for i, s in subjects_group:
            for e in s.groupby(['episode_label']):
                d = self.splitEpisode(e[1])
                if d is not None:
                    X_all.append(d[0])
                    y_all.append(d[1])

        X_all = np.vstack(X_all)
        y_all = np.vstack(y_all)

        if self.training:
            y_last = np.squeeze(y_all[:, -1])
            failures_idx = np.argwhere(y_last == 0)
            success_idx = np.argwhere(y_last == 1)
            np.random.shuffle(success_idx)

            idx_data = np.squeeze(np.concatenate([failures_idx, success_idx[:int(len(success_idx) * .6)]]))

            return X_all[idx_data], y_all[idx_data]
        return X_all, y_all


    def splitEpisode(self, v):
        min_seq = self.n_window
        X_sequence = []
        y_sequence = []


        if len(v) > min_seq :
            for i in range(len(v)//self.n_stride):
                stop = (len(v) - (self.n_stride * i))
                features = v.iloc[stop-min_seq:stop][self.columns].values
                label = v.iloc[stop-min_seq:stop][y_episode_column].values
                if len(features) != min_seq:
                    break
                X_sequence.append(features)
                y_sequence.append(label)

            # the first data
            # features = v.iloc[:min_seq][self.columns].values
            # label = v.iloc[:min_seq][y_episode_column].values
            # X_sequence.append(features)
            # y_sequence.append(label)

            # for i in range(0, 1):
            #     features = v.iloc[-min_seq:][x_columns].values
            #     label = v.iloc[-min_seq:][y_episode_column].values
            #     X_sequence.append(features)
            #     y_sequence.append(label)

            X_sequence = np.asarray(X_sequence)
            y_sequence = np.asarray(y_sequence)
            return X_sequence, y_sequence
        else:
            return None


if __name__ == '__main__':
    path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\single_episode_features.pkl"

    df = pd.read_csv(single_summary_path)
    df = df[(df["norm_score"] > 0.5) & (df["Tobii_percentage"] > 65)]
    features_reader = SequentialFeaturesReader(path, df["Subject1"].values, n_window=4)

    features_reader.getAllData()
