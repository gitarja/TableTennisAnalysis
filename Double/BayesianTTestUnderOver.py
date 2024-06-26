import numpy as np

from Validation.CrossValidation import SubjectCrossValidation, DoubleSubjectCrossValidation
from Double.GlobalFeaturesReader import GlobalFeaturesReader, GlobalDoubleFeaturesReader
import pandas as pd
from sklearn.covariance import EllipticEnvelope
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

np.random.seed(1945)  # For Replicability


def determineLabels(X, y):
    labels = np.ones_like(y)
    # select the outlier and inlier
    cov = EllipticEnvelope(random_state=0, contamination=0.35).fit(X)
    inlier_idx = np.argwhere(cov.predict(X) == 1).flatten()
    outlier_idx = cov.predict(X) == -1

    # set inlier and outlier
    X_inlier = X[inlier_idx]
    y_inlier = y[inlier_idx]

    reg = LinearRegression().fit(X_inlier, y_inlier)

    preds = reg.predict(X)

    labels[(outlier_idx == True) & (preds > y)] = 2  # it predicts greater than the actual
    labels[(outlier_idx == True) & (preds < y)] = 3  # it predicts lower than the actual

    # plt.scatter(X_inlier, y_inlier)
    # plt.scatter(X[labels == 2], y[labels == 2], color="red")
    # plt.scatter(X[labels == 3], y[labels == 3], color="green")
    # plt.plot(X, reg.predict(X), color="blue", linewidth=3)
    # plt.show()
    return labels


def determineLabelsLinearReg(X, y):
    labels = np.ones_like(y)
    reg = LinearRegression().fit(X, y)

    preds = reg.predict(X)

    error = np.abs(y - preds)

    error_th = np.percentile(error, 65)

    labels[(error > error_th) & (preds > y)] = 2
    labels[(error > error_th) & (preds < y)] = 3

    plt.scatter(X[labels == 1], y[labels == 1])
    plt.scatter(X[labels == 2], y[labels == 2], color="red")
    plt.scatter(X[labels == 3], y[labels == 3], color="green")
    plt.plot(X, reg.predict(X), color="blue", linewidth=3)
    plt.show()
    return labels


if __name__ == '__main__':
    rng = np.random.default_rng(seed=42)

    # load single and double data
    single_fr = SubjectCrossValidation()
    double_fr = DoubleSubjectCrossValidation()
    fr = GlobalFeaturesReader(single_fr.getSummary(), double_fr.getSummary())
    X, y, group_label = fr.getSingleDoubleFeatures(log_scale=True, col="max_seq")

    X = np.average(X, axis=-1, keepdims=True)

    labels = determineLabelsLinearReg(X, y)
    inlier_idx = np.argwhere(labels == 1).flatten()
    over_idx = np.argwhere(labels == 2).flatten()
    under_idx = np.argwhere(labels == 3).flatten()

    inlier_group = group_label[inlier_idx]
    over_group = group_label[over_idx]
    under_group = group_label[under_idx]

    # load data
    path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\double_episode_features.pkl"

    # control group
    control_reader = GlobalDoubleFeaturesReader(file_path=path, include_subjects=inlier_group, exclude_failure=False)
    control_features = control_reader.getGlobalFeatures(group_label="control")

    # overestimated group
    over_reader = GlobalDoubleFeaturesReader(file_path=path, include_subjects=over_group, exclude_failure=False)
    over_features = over_reader.getGlobalFeatures(group_label="over")

    # underestimated group
    under_reader = GlobalDoubleFeaturesReader(file_path=path, include_subjects=under_group, exclude_failure=False)
    under_features = under_reader.getGlobalFeatures(group_label="under")

    print(control_features.shape)
    print(over_features.shape)
    print(under_features.shape)
    indv = pd.concat([control_features, over_features, under_features]).reset_index()

    # One Hot Encode Data
    dummies = pd.get_dummies(indv.group)
    # dummies.columns = ['control','over','under']
    df = indv.join(dummies)

    mu_m = df.hitter_fx_duration.mean()
    mu_s = df.hitter_fx_duration.std() * 2

    control_values = control_features["hitter_fx_duration"].values
    under_values = under_features["hitter_fx_duration"].values
    over_values = over_features["hitter_fx_duration"].values
    with pm.Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
        # Define priors
        control_mean = pm.Normal('control_mean', mu=mu_m, sigma=mu_s)
        over_mean = pm.Normal('over_mean', mu=mu_m, sigma=mu_s)
        under_mean = pm.Normal('under_mean', mu=mu_m, sigma=mu_s)

        # Define STD
        control_sigma = pm.Uniform("control_sigma", lower=0.01, upper=0.5)
        over_sigma = pm.Uniform("over_sigma", lower=0.01, upper=0.5)
        under_sigma = pm.Uniform("under_sigma", lower=0.01, upper=0.5)

        # Define nu
        nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
        nu = pm.Deterministic("nu", nu_minus_one + 1)
        nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))

        # Likelihood group
        lambda_1 = control_sigma ** -2
        lambda_2 = under_sigma ** -2
        labda_3 = over_sigma ** -2
        control = pm.StudentT("control", nu=nu, mu=control_mean, lam=lambda_1, observed=control_values)
        under = pm.StudentT("under", nu=nu, mu=under_mean, lam=lambda_2, observed=under_values)
        over = pm.StudentT("over", nu=nu, mu=over_mean, lam=labda_3, observed=over_values)

        # compute diff means
        diff_means_control_under = pm.Deterministic('diff_means_control_under', control_mean - under_mean)
        diff_means_control_over = pm.Deterministic('diff_means_control_over', control_mean - over_mean)
        diff_means_under_over = pm.Deterministic('diff_means_under_over', under_mean - over_mean)

        # compute diff std
        diff_stds_control_under = pm.Deterministic("diff_stds_control_under", control_sigma - under_sigma)
        diff_stds_control_over = pm.Deterministic("diff_stds_control_over", control_sigma - over_sigma)
        diff_stds_under_over = pm.Deterministic("diff_stds_under_over", over_sigma - under_sigma)

        # compute effect size
        effect_control_under = pm.Deterministic(
            "effect_control_under", diff_means_control_under / np.sqrt((control_sigma ** 2 + under_sigma ** 2) / 2)
        )
        effect_control_over = pm.Deterministic(
            "effect_control_over", diff_means_control_over/ np.sqrt((control_sigma ** 2 + over_sigma ** 2) / 2)
        )
        effect_under_over = pm.Deterministic(
            "effect_under_over", diff_means_under_over / np.sqrt((over_sigma ** 2 + under_sigma ** 2) / 2)
        )

        trace = pm.sample(4000, tune=1000,  cores=4, chains=5)

    az.plot_forest(trace, var_names=["control_mean", "over_mean", "under_mean"])
    plt.show()

