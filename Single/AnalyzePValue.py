import pickle
import pandas as pd
from Conf import single_results_path
from statsmodels.stats.multitest import fdrcorrection
import glob
import numpy as np

df_pvlues_list = []
for file in glob.glob(single_results_path + "single_statistic_all*"):
    print(file)
    with open(file, 'rb') as f:

        df_pvlues_list = df_pvlues_list + pickle.load(f)


# def normalizedP(values):
#
#     normalized = -2 * np.sum(np.log(values))
#     return normalized
df_pvalues = pd.concat(df_pvlues_list)
summarized_df = df_pvalues.groupby(["features_name"]).mean()

p_val = summarized_df.loc[:, "P-val"].values
corrected_pval = fdrcorrection(p_val, alpha=0.05, is_sorted=False, method="indep")
summarized_df["corrected"] = corrected_pval[-1]
summarized_df["accepted"] = corrected_pval[0]
summarized_df.to_csv(single_results_path + "summarized_pvalues-mean.csv")