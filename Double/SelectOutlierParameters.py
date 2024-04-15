import numpy as np

from Validation.CrossValidation import SubjectCrossValidation, DoubleSubjectCrossValidation
from Double.GlobalFeaturesReader import GlobalFeaturesReader
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

def trainAndEvaluate(X, y):
    model = LinearRegression().fit(X, y)
    preds = model.predict(X)

    return model, preds



single_fr = SubjectCrossValidation()
double_fr = DoubleSubjectCrossValidation()

fr = GlobalFeaturesReader(single_fr.getSummary(), double_fr.getSummary())

X, y, _ = fr.getSingleDoubleFeatures(log_scale=True, col="max_seq")

X = np.average(X, axis=-1, keepdims=True)


# pca = PCA(n_components=2)
#
# X_plot = pca.fit_transform(X)

_, preds1 = trainAndEvaluate(X, y)
MAP_score_origin = mean_absolute_percentage_error(y, preds1)
MSE_score_origin = mean_squared_error(y, preds1, squared=True)
print("%f, %f" % (MSE_score_origin, MAP_score_origin))

for c in np.arange(0.05, 0.55, 0.05):
    cov = EllipticEnvelope(random_state=0, contamination=c).fit(X)
    # cov = IsolationForest().fit(X)
    inlier_idx = np.argwhere(cov.predict(X) == 1).flatten()
    outlier_idx = np.argwhere(cov.predict(X) == -1).flatten()

    plt.scatter(X[inlier_idx], y[inlier_idx])
    plt.scatter(X[outlier_idx], y[outlier_idx], color="red")
    plt.show()

    _, preds2 = trainAndEvaluate(X[inlier_idx], y[inlier_idx])
    MAP_score = mean_absolute_percentage_error(y[inlier_idx], preds2)
    MSE_score = mean_squared_error(y[inlier_idx], preds2, squared=True)

    print("%f, %f" % (MSE_score, MAP_score))

