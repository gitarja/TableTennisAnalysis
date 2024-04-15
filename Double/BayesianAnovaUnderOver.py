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

    return labels


def determineLabelsLinearReg(X, y):
    labels = np.ones_like(y)
    reg = LinearRegression().fit(X, y)

    preds = reg.predict(X)

    error = np.abs(y - preds)

    error_th = np.percentile(error, 65)

    labels[(error > error_th) & (preds > y)] = 2
    labels[(error > error_th) & (preds < y)] = 3

    return labels


if __name__ == '__main__':
    rng = np.random.default_rng(seed=42)

    # load single and double data
    single_fr = SubjectCrossValidation()
    double_fr = DoubleSubjectCrossValidation()
    fr = GlobalFeaturesReader(single_fr.getSummary(), double_fr.getSummary())
    X, y, group_label = fr.getSingleDoubleFeatures()

    X = np.average(X, axis=-1, keepdims=True)

    labels = determineLabels(X, y)
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

    with pm.Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
        # Define priors
        sigma = pm.Uniform("sigma", lower=0.01, upper=200)
        control = pm.Normal('control', mu=mu_m, sigma=mu_s)
        over = pm.Normal('over', mu=mu_m, sigma=mu_s)
        under = pm.Normal('under', mu=mu_m, sigma=mu_s)

        # Define likelihood
        likelihood = pm.Normal('likelihood',
                               mu=control * df['control'] + under * df['under'] + over * df['over'],
                               sigma=sigma,
                               observed=df.hitter_fx_duration)

        # pm.model_to_graphviz(model).view()

        # Inference!
        trace = pm.sample(4000, cores=4, chains=4)  # draw 4000 posterior samples using NUTS sampling


    trace_post = trace["posterior"]

    # Get posterior samples for the parameter of interest
    # posterior_samples = np.concatenate([trace_post['control'].data.flatten()])
    # credible_interval = np.percentile(posterior_samples, [5, 95.0])
    # print("Credible Interval (95%):", credible_interval)

    alpha = 0.05
    l = len(trace_post['control'].data.flatten())
    low_bound = int(alpha / 2 * l)
    high_bound = int((1 - (alpha / 2)) * l)

    fig, ax = plt.subplots(figsize=(12, 8))
    for group, color in zip(['control', 'over', 'under'], ['#95d0fc', '#ff796c', '#380282']):
        # Estimate KDE
        kde = stats.gaussian_kde(trace_post[group].data.flatten())
        # plot complete kde curve as line
        pos = np.linspace(trace_post[group].min(), trace_post[group].max(), 101)
        plt.plot(pos, kde(pos), color=color, label='{0} KDE'.format(group))
        # Set shading bounds
        low = np.sort(trace_post[group].data.flatten())[low_bound]
        high = np.sort(trace_post[group].data.flatten())[high_bound]
        # plot shaded kde
        shade = np.linspace(low, high, 101)
        plt.fill_between(shade, kde(shade), alpha=0.3, color=color, label="{0} 95% HPD Score".format(group))
    plt.legend()
    plt.xlabel("Hitter pursuit duration")
    plt.show()
