from sklearn import svm
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

scaler = preprocessing.RobustScaler()
Xtr = scaler.fit_transform(Xtr)
Xte = scaler.transform(Xte)


transformer = KernelPCA(n_components=500, kernel='poly')
Xtr = transformer.fit_transform(Xtr)
Xte = transformer.transform(Xte)


model = KNeighborsClassifier()

# param_grid = {
#  'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
#  'gamma' : ['scale', 'auto'],
#  'decision_function_shape' : ['ovo', 'ovr']
# }

# grid_model = model_selection.RandomizedSearchCV(model,
#                                                 param_grid=param_grid,
#                                                 cv=10,
#                                                 scoring = 'balanced_accuracy',
#                                                 verbose=3,
#                                                 n_jobs=1)


distributions = dict(n_neighbors=stats.uniform(loc=2, scale=50),
                      weights=['uniform', 'distance'],
                      algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'],
                      leaf_size=stats.uniform(loc=20, scale=40),
                      p=[1,2])

grid_model = model_selection.RandomizedSearchCV(model,
                                                distributions,
                                                random_state=0,
                                                scoring = 'balanced_accuracy',
                                                verbose=3,
                                                n_jobs=-1)

print("Fitting model...")

grid_model.fit(Xtr, np.ravel(ytr))
print(grid_model.score(Xte, yte))
print(grid_model.best_params_)