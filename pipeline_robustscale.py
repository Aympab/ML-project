from random import uniform
from utils import *
from sklearn import cluster
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import KernelPCA
from scipy import stats

X, y, X_test, X_valid = load_data("starting_kit/data")

Xtr, Xte, ytr, yte = model_selection.train_test_split(X, y, 
                                                      test_size=0.2, 
                                                      random_state=0)

pipe = Pipeline([('scaler', RobustScaler()),
                ('reduction', KernelPCA()),
                ('simple_tree', KNeighborsClassifier(algorithm='brute',
                                                        leaf_size=50,
                                                        n_neighbors=10,
                                                        weights='uniform'))
                ])

grid_model = model_selection.RandomizedSearchCV(pipe,
                            #param_grid = {'reduction__n_components':[0.5, 0.6]},
                            param_distributions = {'reduction__n_components': stats.uniform(loc=100, scale=800),
                                          'reduction__kernel':['linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'],
                                          'reduction__eigen_solver' : ['randomized']
                                          },
                            scoring = 'balanced_accuracy',
                            cv = 5,
                            verbose=2,
                            n_jobs=1)


grid_model.fit(Xtr, ytr, )
print(grid_model.score(Xte, yte))
print(grid_model.best_params_)