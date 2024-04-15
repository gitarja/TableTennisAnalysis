import pandas as pd
from Conf import double_summary_path, single_summary_path
import numpy as np
from Conf import x_episode_columns, x_double_features_column
from sklearn.impute import KNNImputer
class DoubleDataReader:


    def __init__(self, features_pkl=None):
        df_summary = pd.read_csv(double_summary_path)
        df_summary = df_summary[(df_summary["norm_score"] > 0.55) & (df_summary["Tobii_percentage"] > 65)]

        df = pd.read_pickle(features_pkl)

        df = df.loc[
             (df["success"] != -1) & (df["pair_idx"] != -1) & (df["session_id"].isin(df_summary["file_name"].values)),
             :]

        df = df.loc[(df["session_id"] != "2022-12-19_A_T06") | (
                df["session_id"] != "2023-02-15_M_T01")]  # session excluded, equipments fail

        self.df = df



    def getGroup(self, min_n=15):

        data = self.df

        # input missing values
        imputer = KNNImputer(n_neighbors=5)
        data.loc[:, x_double_features_column] = imputer.fit_transform(
            data.loc[:, x_double_features_column])

        selected_groups = data.groupby(["session_id", "episode_label"]).filter(lambda x: len(x) >= min_n*2)

        grouped_episodes = selected_groups.groupby(["session_id", "episode_label"])


        return grouped_episodes


class SingleFeatures:
    def __init__(self, features_pkl=None):

        df_summary = pd.read_csv(single_summary_path)
        df_summary = df_summary[(df_summary["norm_score"] > 0.55) & (df_summary["Tobii_percentage"] > 65)]

        df = pd.read_pickle(features_pkl)

        df = df.loc[
             (df["success"] != -1) & (df["id_subject"].isin(df_summary["Subject1"].values)),
             :]

        self.df = df



    def getGroup(self, min_n = 15):

        data = self.df

        # input missing values
        imputer = KNNImputer(n_neighbors=5)
        data.loc[:, x_episode_columns] = imputer.fit_transform(
            data.loc[:, x_episode_columns])

        selected_groups = data.groupby(["id_subject", "episode_label"]).filter(lambda x: len(x) >= min_n)

        grouped_episodes = selected_groups.groupby(["id_subject", "episode_label"])


        return grouped_episodes


