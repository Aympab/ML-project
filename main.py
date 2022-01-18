from utils import *

X, y, X_test, X_valid = load_data("starting_kit/data")
################################################################################
##############################  MODEL  #########################################
################################################################################

from sklearn import ensemble
from sklearn import model_selection, metrics

model = ensemble.BaggingClassifier()

param_grid = {
    'max_features': [0.5],
    'max_samples': [0.7],
    'n_estimators': [10],
}

cv_model = model_selection.GridSearchCV(model,
                                        param_grid=param_grid,
                                        cv=10,
                                        verbose=3,
                                        n_jobs=-1)



print("score : ", cv_model.score(X, y))
print("Best param : ", cv_model.best_params_)


################################################################################
##############################  MODEL  #########################################
################################################################################
print("Predicting...")
cv_model.fit(X, y)

print("Submitting...")
submit_model(model, X_test, X_valid)