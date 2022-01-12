import numpy as np
from zipfile import ZipFile
from sklearn import ensemble
from sklearn import model_selection
from sklearn import preprocessing

print("Loading data...")

X = np.loadtxt("data/protein_train.data")
y = np.loadtxt("data/protein_train.solution")

X_test = np.loadtxt("data/protein_test.data")
X_valid = np.loadtxt("data/protein_valid.data")

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)

bag = ensemble.RandomForestClassifier()

param_grid = {
 'max_features': ['sqrt', 'log2', 0.15],
 'max_depth' : [10,20,50],
 'min_weight_fraction_leaf' : [0.0,0.1,0.01,0.5],
 'n_estimators' : [100,150,200],
 'criterion' : ['gini', 'entropy'],
}

#param_grid = {
#    'max_features': [0.15],
#    'max_depth' : [25],
#    'min_weight_fraction_leaf' : [0.1],
#    'n_estimators' : [150],
#    'criterion' : ['entropy'],
#}

cv_bagging = model_selection.GridSearchCV(bag, param_grid=param_grid, cv=10, verbose=3, n_jobs=-1)

#cv_bagging = model_selection.cross_validate(bag, X, y=y, cv=10, verbose=3, n_jobs=-1)

print("Fitting model...")

cv_bagging.fit(X, y)

print("score : ", cv_bagging.score(X, y))

print("Predicting...")

y_test = cv_bagging.predict(X_test)
y_valid = cv_bagging.predict(X_valid)

np.savetxt("protein_test.predict", y_test, fmt="%d")
np.savetxt("protein_valid.predict", y_valid, fmt="%d")
zip_obj = ZipFile('submission.zip', 'w')
zip_obj.write("protein_test.predict")
zip_obj.write("protein_valid.predict")

zip_obj.close()
