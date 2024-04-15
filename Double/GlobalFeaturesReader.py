import numpy as np
from scipy.special import logit, expit
import pandas as pd

class GlobalFeaturesReader:


    def __init__(self, single_summary, double_summary):

        self.single = single_summary
        self.double = double_summary



    def getSingleDoubleFeatures(self, log_scale=True, col="skill"):
        X = []
        y = []
        group_label = []
        for index, row in self.double.iterrows():
            subject1 = row["Subject1"]
            subject2 = row["Subject2"]

            subject1_skill = self.single.loc[self.single["Subject1"] == subject1][[col]].values
            subject2_skill = self.single.loc[self.single["Subject1"] == subject2][[col]].values


            pair_skill = row[col]

            group_name = row["file_name"]

            if (len(subject1_skill) > 0) & (len(subject2_skill) > 0):
                X.append(np.concatenate([subject1_skill[0], subject2_skill[0]]))
                y.append(pair_skill)
                group_label.append(group_name)


        if log_scale:
            return np.log2(np.vstack(X)), np.log2(np.asarray(y)), np.asarray(group_label)

        else:
            return np.vstack(X), np.asarray(y), np.asarray(group_label)


class GlobalDoubleFeaturesReader:

    def __init__(self, file_path="", include_subjects=["test"], exclude_failure=True):
        df = pd.read_pickle(file_path)

        # select subjects subjects
        df = df.loc[df["session_id"].isin(include_subjects), :]



        if exclude_failure:
            df = df.loc[(df["success"] != 0)| (df["success"] != -1)]
        else:
            df = df.loc[df["success"] != -1]

        self.df = df



    def getGlobalFeatures(self, group_label="control"):

        group_df = self.df.groupby(['session_id'])
        receiver_al = []
        receiver_pursuit_du = []
        hitter_al = []
        hitter_putsuit_du = []
        for name, group in group_df:

            # percentage of AL in phase 2
            receiver_al.append(group["receiver_pr_p2_al"].mean())
            hitter_al.append(group["hitter_pr_p2_al"].mean())
            # duration of pursuit
            receiver_pursuit_du.append(group["receiver_pr_p3_fx_duration"].mean())
            hitter_putsuit_du.append(group["hitter_pr_p3_fx"].mean())



        fetures_summary = {
            "receiver_al_percentage": np.asarray(receiver_al),
            "receiver_fx_duration": np.asarray(receiver_pursuit_du),
            "hitter_al_percentage": np.asarray(hitter_al),
            "hitter_fx_duration": np.asarray(hitter_putsuit_du),
            "group": group_label
        }



        return  pd.DataFrame(fetures_summary)