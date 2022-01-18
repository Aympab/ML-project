import numpy as np
from zipfile import ZipFile
from sklearn import ensemble
from sklearn import model_selection
from sklearn import preprocessing, neighbors, decomposition, linear_model

print("Loading data...")

X = np.loadtxt("data/protein_train.data")
y = np.loadtxt("data/protein_train.solution")

X_test = np.loadtxt("data/protein_test.data")
X_valid = np.loadtxt("data/protein_valid.data")

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)


pca = decomposition.PCA(n_components=500)
pca.fit(X)
X = pca.transform(X)
X_test = pca.transform(X_test)
X_valid = pca.transform(X_valid)

log_reg = linear_model.RidgeClassifier()


param_grid = {
 'alpha': [2.0],
 'tol' : [1e-10, 1e-5, 1e-3, 1e-4,],
}

cv_log_reg = model_selection.GridSearchCV(log_reg, param_grid=param_grid, cv=10, verbose=3, n_jobs=-1)


print("Fitting model...")

cv_log_reg.fit(X, y)

print("score : ", cv_log_reg.score(X, y))
print(cv_log_reg.best_params_)
print("Predicting...")

y_test = cv_log_reg.predict(X_test)
y_valid = cv_log_reg.predict(X_valid)

np.savetxt("protein_test.predict", y_test, fmt="%d")
np.savetxt("protein_valid.predict", y_valid, fmt="%d")
zip_obj = ZipFile('submission.zip', 'w')
zip_obj.write("protein_test.predict")
zip_obj.write("protein_valid.predict")

zip_obj.close()
