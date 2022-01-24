import numpy as np
from sklearn.cluster import FeatureAgglomeration
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from utils import *
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import norm

print("###BEGIN###")
print("Loading data...")
X, y, X_test, X_valid = load_data("data")

################################################################################
##############################  SCALING  #######################################
################################################################################
print("Scaling...")
scaler = RobustScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)

################################################################################
##############################  REDUCTION  #####################################
################################################################################
print("Reduction...")
transformer = FeatureAgglomeration(n_clusters=600)
X = transformer.fit_transform(X)
X_test = transformer.transform(X_test)
X_valid = transformer.transform(X_valid)

################################################################################
##############################  MODELS  ########################################
################################################################################
mlp = MLPClassifier(activation='logistic', learning_rate='adaptive', alpha=0.01)

model = MLPClassifier(max_iter=400)

# param_grid = {
#  'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
#  'gamma' : ['scale', 'auto'],
#  'decision_function_shape' : ['ovo', 'ovr']
# }


distributions = dict(alpha=norm(loc=0.01, scale=0.20),
                     learning_rate=['adaptive'],
                     activation=['logistic'],
                     solver=['lbfgs', 'sgd', 'adam'],
                     learning_rate_init=norm(loc=0.001, scale=0.02),
                     early_stopping=[True, False],
                     validation_fraction=norm(loc=0.1, scale=0.1)
                     )

grid_model = RandomizedSearchCV(model,
                                distributions,
                                cv=5,
                                scoring = 'balanced_accuracy',
                                verbose=3,
                                n_jobs=-1,
                                random_state=0)

print("Fitting model...")

grid_model.fit(X, np.ravel(y))
print(grid_model.score(X, y))
print(grid_model.best_params_)
print("Submitting...")
submit_model(grid_model, X_test, X_valid)
