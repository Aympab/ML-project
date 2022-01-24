"""
==========================
Plotting Validation Curves
==========================

In this plot you can see the training scores and validation scores of an SVM
for different values of the kernel parameter gamma. For very low values of
gamma, you can see that both the training score and the validation score are
low. This is called underfitting. Medium values of gamma will result in high
values for both scores, i.e. the classifier is performing fairly well. If gamma
is too high, the classifier will overfit, which means that the training score
is good but the validation score is poor.

"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import FeatureAgglomeration
from sklearn.neural_network import MLPClassifier
from utils import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


# X, y = load_digits(return_X_y=True)
# X, y = X[subset_mask], y[subset_mask]

X, y, X_test, X_valid = load_data("starting_kit/data") 

trunc = 10000
X = X[:trunc]
y = y[:trunc]

scaler = RobustScaler()
X = scaler.fit_transform(X)

# subset_mask = np.isin(y, [0, 1])  # binary classification: 1 vs 2X

#transformer = FeatureAgglomeration(n_clusters=650)
#X = transformer.fit_transform(X)

#estimator = MLPClassifier()
#estimator = MLPClassifier(activation='logistic', learning_rate='adaptive', alpha=0.01)

#scaler = FeatureAgglomeration()

pipe = Pipeline([('reduction', KernelPCA(kernel='linear')),
                 ('simple_tree', KNeighborsClassifier(algorithm='brute',
                                                      n_neighbors=10,
                                                      p=1))
                            ])


param_range = np.linspace(10, 950, 4)

train_scores, test_scores = validation_curve(
    pipe,
    X,
    np.ravel(y),
    cv=4,
    param_name="reduction__n_components",
    param_range=param_range,
    scoring="balanced_accuracy",
    n_jobs=2,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve")
plt.xlabel("n_components")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(
    param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw
)
plt.fill_between(
    param_range,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2,
    color="darkorange",
    lw=lw,
)
plt.semilogx(
    param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw
)
plt.fill_between(
    param_range,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.2,
    color="navy",
    lw=lw,
)
plt.legend(loc="best")
plt.savefig("validation_curve.png")
# plt.show()
