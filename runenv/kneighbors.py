import numpy as np
from zipfile import ZipFile
from sklearn import ensemble
from sklearn import model_selection
from sklearn import preprocessing, neighbors, decomposition

print("Loading data...")

X = np.loadtxt("../data/protein_train.data")
y = np.loadtxt("../data/protein_train.solution")

X_test = np.loadtxt("../data/protein_test.data")
X_valid = np.loadtxt("../data/protein_valid.data")

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)


pca = decomposition.PCA(n_components=750)
pca.fit(X)
X = pca.transform(X)
X_test = pca.transform(X_test)
X_valid = pca.transform(X_valid)


kn = neighbors.KNeighborsClassifier()

# param_grid = {
#  'n_neighbors': [5],
#  'weights' : ['distance'],
#  'algorithm' : [ 'ball_tree'],
#  'leaf_size' :[10]
# }

param_grid = {
 'n_neighbors': [5, 2, 10, 50],
 'weights' : ['distance'],
 'algorithm' : [ 'ball_tree'],
 'leaf_size' :[10]
}
# algorithm=ball_tree, leaf_size=50, n_neighbors=5, p=1, weights=distance, score=0.922, total=10.3min

cv_knn = model_selection.GridSearchCV(kn, param_grid=param_grid, cv=10, verbose=3, n_jobs=-1)

#cv_knn = model_selection.cross_validate(bag, X, y=y, cv=10, verbose=3, n_jobs=-1)

print("Fitting model...")

cv_knn.fit(X, y)

print("score : ", cv_knn.score(X, y))
print(cv_knn.best_params_)
print("Predicting...")

y_test = cv_knn.predict(X_test)
y_valid = cv_knn.predict(X_valid)

np.savetxt("protein_test.predict", y_test, fmt="%d")
np.savetxt("protein_valid.predict", y_valid, fmt="%d")
zip_obj = ZipFile('submission_knn_pca750_neigh5.zip', 'w')
zip_obj.write("protein_test.predict")
zip_obj.write("protein_valid.predict")

zip_obj.close()
