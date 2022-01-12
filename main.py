import numpy as np
from zipfile import ZipFile
from sklearn import ensemble
from sklearn import model_selection

print("Loading data...")

X = np.loadtxt("data/protein_train.data")
y = np.loadtxt("data/protein_train.solution")

X_test = np.loadtxt("data/protein_test.data")
X_valid = np.loadtxt("data/protein_valid.data")

bag = ensemble.BaggingClassifier()
param_grid = {
    'max_features': [0.8],
    'max_samples': [0.8],
    'n_estimators': [10],
}
cv_bagging = model_selection.GridSearchCV(bag, param_grid=param_grid, cv=10, verbose=3, n_jobs=-1)

print("Fitting model...")

cv_bagging.fit(X, y)


print("Predicting...")

y_test = cv_bagging.predict(X_test)
y_valid = cv_bagging.predict(X_valid)

np.savetxt("protein_test.predict", y_test, fmt="%d")
np.savetxt("protein_valid.predict", y_valid, fmt="%d")
zip_obj = ZipFile('submission.zip', 'w')
zip_obj.write("protein_test.predict")
zip_obj.write("protein_valid.predict")

zip_obj.close()