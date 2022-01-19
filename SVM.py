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

X, y, X_test, X_valid = load_data("data")

# Xtr, Xte, ytr, yte = model_selection.train_test_split(X, y, 
#                                                       test_size=0.2, 
#                                                       random_state=0)

scaler = preprocessing.RobustScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)

transformer = KernelPCA(n_components=100, kernel='poly')
X = transformer.fit_transform(X)
X_test = transformer.transform(X_test)
X_valid = transformer.transform(X_valid)

model = svm.SVC()

# param_grid = {
#  'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
#  'gamma' : ['scale', 'auto'],
#  'decision_function_shape' : ['ovo', 'ovr']
# }

grid_model = model_selection.GridSearchCV(model,
                                          param_grid={},
                                          cv=10,
                                          scoring = 'balanced_accuracy',
                                          verbose=3,
                                          n_jobs=-1)

print("Fitting model...")

grid_model.fit(X, y)
print(grid_model.score(X, y))
print(grid_model.best_params_)
print("Submitting...")
submit_model(grid_model, X_test, X_valid)

# grid_model.fit(Xtr, np.ravel(ytr))
# print(grid_model.score(Xte, yte))
# print(grid_model.best_params_)