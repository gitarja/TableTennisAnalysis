import numpy as np
import pandas as pd
from sklearn import linear_model
import scipy.stats as stat
from tabulate import tabulate

pd.options.display.float_format = '{:,.3f}'.format
class LogisticRegression(linear_model.LogisticRegression):

    def __init__(self, *args, **kwargs):  # ,**kwargs):
        self.model = linear_model.LogisticRegression(*args, **kwargs)  # ,**args)
        if 'fit_intercept' in kwargs.keys():
            self._fit_intercept = kwargs['fit_intercept']

    def fit(self, X, y):
        eps = 1e-10

        self.model.fit(X, y)
        #### Get p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))


        if self._fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        if self._fit_intercept:
            self.coef = np.column_stack((self.model.intercept_, self.model.coef_))[0]
        else:
            self.coef = self.model.coef_[0]

        d = np.tile(denom, (X.shape[1], 1)).T
        F_ij = np.dot((X / d).T, X)
        F_ij = F_ij + np.eye(F_ij.shape[0])*eps## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij)  ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_score = (self.coef / sigma_estimates)  # z-score for each model coefficient

        p_vals = [stat.norm.sf(abs(i)) * 2 for i in z_score]  ### two tailed test for p-values

        self.z_scores =z_score
        self.p_values = p_vals
        self.sigma_estimates = sigma_estimates
        self.F_ij = F_ij


    def printSummary(self, X_columns, print_df=False):


        X_columns = ["intercept"] + X_columns

        data = {"features_name": X_columns, "Coeff": self.coef, "Z-score": self.z_scores, "P-val": self.p_values}

        df = pd.DataFrame(data)

        if print_df:
            print(tabulate(df.round(decimals=3), headers='keys', tablefmt='psql'))

        return df
