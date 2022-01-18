from utils import *

X, y, X_test, X_valid = load_data("starting_kit/data")
################################################################################
##############################  MODEL  #########################################
################################################################################

<<<<<<< HEAD
pca = decomposition.PCA(n_components=500)
pca.fit(X)
X = pca.transform(X)
X_test = pca.transform(X_test)
X_valid = pca.transform(X_valid)
=======
from sklearn import ensemble
from sklearn import model_selection, metrics
>>>>>>> 946ed9dd6e0d28fd43ca5614755cc66b27dee88e

model = ensemble.BaggingClassifier()

param_grid = {
<<<<<<< HEAD
   'max_features': [0.15, 1.0, 0.5, 0.2],
   'max_depth' : [20],
   'min_weight_fraction_leaf' : [0.0],
   'n_estimators' : [100],
   'criterion' : ['entropy'],
}
#{'criterion': 'entropy', 'max_depth': 20, 'max_features': 0.5, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100}
cv_bagging = model_selection.GridSearchCV(bag, param_grid=param_grid, cv=10, verbose=3, n_jobs=-1)
=======
    'max_features': [0.5],
    'max_samples': [0.7],
    'n_estimators': [10],
}

cv_model = model_selection.GridSearchCV(model,
                                        param_grid=param_grid,
                                        cv=10,
                                        verbose=3,
                                        n_jobs=-1)
>>>>>>> 946ed9dd6e0d28fd43ca5614755cc66b27dee88e



print("score : ", cv_model.score(X, y))
print("Best param : ", cv_model.best_params_)


################################################################################
##############################  MODEL  #########################################
################################################################################
print("Predicting...")
cv_model.fit(X, y)

print("Submitting...")
submit_model(model, X_test, X_valid)