from utils import *
from sklearn import cluster
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import KernelPCA

X, y, X_test, X_valid = load_data("data")

Xtr, Xte, ytr, yte = model_selection.train_test_split(X, y, 
                                                      test_size=0.2, 
                                                      random_state=0)

pipe = Pipeline([('scaler', StandardScaler()),
                ('reduction', KernelPCA()),
                ('simple_tree', KNeighborsClassifier(algorithm='brute',
                                                        leaf_size=50,
                                                        n_neighbors=10,
                                                        weights='uniform'))
                ])

grid_model = model_selection.GridSearchCV(pipe,
                            #param_grid = {'reduction__n_components':[0.5, 0.6]},
                            param_grid = {'reduction__n_components':[100, 250, 400, 500, 600, 750, 900],
                                          'reduction__kernel':['linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'],
                                          'reduction__eigen_solver' : ['randomized']
                                          },
                            scoring = 'balanced_accuracy',
                            cv = 5,
                            verbose=2,
                            n_jobs=-1)


grid_model.fit(Xtr, ytr, )
print(grid_model.score(Xte, yte))
print(grid_model.best_params_)