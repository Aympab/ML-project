import numpy as np
from zipfile import ZipFile
from sklearn import ensemble
from sklearn import model_selection
from sklearn import preprocessing, neighbors, neural_network, gaussian_process

print("Loading data...")

X = np.loadtxt("data/protein_train.data")
y = np.loadtxt("data/protein_train.solution")

X_test = np.loadtxt("data/protein_test.data")
X_valid = np.loadtxt("data/protein_valid.data")

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)

gp = gaussian_process.GaussianProcessClassifier()

param_grid = {
 'multi_class': ['one_vs_rest', 'one_vs_one'],
 'copy_X_train' : [False],
}


cv_gp = model_selection.GridSearchCV(gp, param_grid=param_grid, cv=10, verbose=3, n_jobs=-1)

print("Fitting model...")


cv_gp.fit(X, y)

print("score : ", cv_gp.score(X, y))
print(cv_gp.best_params_)

print("Predicting...")

y_test = cv_gp.predict(X_test)
y_valid = cv_gp.predict(X_valid)

np.savetxt("gauss_protein_test.predict", y_test, fmt="%d")
np.savetxt("gauss_protein_valid.predict", y_valid, fmt="%d")
zip_obj = ZipFile('submission_gauss.zip', 'w')
zip_obj.write("gauss_protein_test.predict")
zip_obj.write("gauss_protein_valid.predict")

zip_obj.close()
