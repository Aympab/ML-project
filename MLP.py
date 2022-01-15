import numpy as np
from zipfile import ZipFile
from sklearn import ensemble
from sklearn import model_selection
from sklearn import preprocessing, neighbors, neural_network

print("Loading data...")

X = np.loadtxt("data/protein_train.data")
y = np.loadtxt("data/protein_train.solution")

X_test = np.loadtxt("data/protein_test.data")
X_valid = np.loadtxt("data/protein_valid.data")

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)

mlp = neural_network.MLPClassifier()

param_grid = {
 'activation': ['identity', 'logistic', 'relu', 'tanh'],
 'learning_rate' : ['constant', 'adaptive', 'invscaling'],
 'alpha' : [0.0001, 0.01],
}

#param_grid = {
#    'max_features': [0.15],
#    'max_depth' : [25],
#    'min_weight_fraction_leaf' : [0.1],
#    'n_estimators' : [150],
#    'criterion' : ['entropy'],
#}

cv_mlp = model_selection.GridSearchCV(mlp, param_grid=param_grid, cv=10, verbose=3, n_jobs=-1)

print("Fitting model...")


cv_mlp.fit(X, y)

print("score : ", cv_mlp.score(X, y))
print(cv_mlp.best_params_)

print("Predicting...")

y_test = cv_mlp.predict(X_test)
y_valid = cv_mlp.predict(X_valid)

np.savetxt("mlp_protein_test.predict", y_test, fmt="%d")
np.savetxt("mlp_protein_valid.predict", y_valid, fmt="%d")
zip_obj = ZipFile('submission_mlp.zip', 'w')
zip_obj.write("mlp_protein_test.predict")
zip_obj.write("mlp_protein_valid.predict")

zip_obj.close()
