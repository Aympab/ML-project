import numpy as np
from zipfile import ZipFile
from sklearn import ensemble
from sklearn import model_selection
from sklearn import preprocessing, neighbors

print("Loading data...")

X = np.loadtxt("data/protein_train.data")
y = np.loadtxt("data/protein_train.solution")

X_test = np.loadtxt("data/protein_test.data")
X_valid = np.loadtxt("data/protein_valid.data")

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)

kn = neighbors.KNeighborsClassifier()

param_grid = {
 'max_features': [0.15, 0.2, 0.3],
 'n_neighbors': [5, 10, 50],
 'weights' : ['uniform', 'distance'],
 'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
 'leaf_size' :[30, 50, 10],
 'p' : [1,2]
}

#param_grid = {
#    'max_features': [0.15],
#    'max_depth' : [25],
#    'min_weight_fraction_leaf' : [0.1],
#    'n_estimators' : [150],
#    'criterion' : ['entropy'],
#}

cv_knn = model_selection.GridSearchCV(kn, param_grid=param_grid, cv=7, verbose=3, n_jobs=-1)

#cv_knn = model_selection.cross_validate(bag, X, y=y, cv=10, verbose=3, n_jobs=-1)

print("Fitting model...")

cv_knn.fit(X, y)

print("score : ", cv_knn.score(X, y))

print("Predicting...")

y_test = cv_knn.predict(X_test)
y_valid = cv_knn.predict(X_valid)

np.savetxt("protein_test.predict", y_test, fmt="%d")
np.savetxt("protein_valid.predict", y_valid, fmt="%d")
zip_obj = ZipFile('submission_knn.zip', 'w')
zip_obj.write("protein_test.predict")
zip_obj.write("protein_valid.predict")

zip_obj.close()
