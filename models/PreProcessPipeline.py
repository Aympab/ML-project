
import numpy as np
import pandas as pd
from utils import *
from sklearn import cluster
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from scipy.stats import uniform

X = pd.read_csv("data/protein_train.data", sep=" ", header=None)
y = pd.read_csv("data/protein_train.solution", sep=" ", header=None)

Xtr, Xte, ytr, yte = model_selection.train_test_split(X, y,
                                                      test_size=0.2,
                                                      random_state=0)

scalers_list = [StandardScaler, RobustScaler, QuantileTransformer]
reduction_list = [KernelPCA, GaussianRandomProjection, FeatureAgglomeration]
params_grid_list = [{'reduction__n_components': uniform(loc=10, scale=900), "reduction__kernel":["linear", "poly", "rbf"]},
                    {'reduction__n_components':uniform(loc=10, scale=900)},
                    {'reduction__n_clusters':uniform(loc=10, scale=900)}
]

for scaler in scalers_list:
    for reduction, params_grid in zip(reduction_list, params_grid_list):
        pipe = Pipeline([('scaler', scaler()),
                        ('reduction', reduction()),
                        ('simple_tree', KNeighborsClassifier(algorithm='brute',
                                                                n_neighbors=10,
                                                                p=1))
                        ])

        grid_model = model_selection.RandomizedSearchCV(
                                    pipe,
                                    params_grid,
                                    scoring = 'balanced_accuracy',
                                    cv = 3,
                                    n_jobs=1,
                                    verbose=1)


        grid_model.fit(Xtr, ytr.values.flatten(), )
        print(grid_model.score(Xte, yte.values.flatten()))
        print(grid_model.best_params_)
