import matplotlib.pyplot as plt
import pandas as pd
from sklearn import ensemble, model_selection

X_data = pd.read_csv("data/protein_train.data", sep=" ", header=None)
y_data = pd.read_csv("data/protein_train.solution", sep=" ", header=None)

X_test_data = pd.read_csv("data/protein_test.data", sep=" ", header=None)
X_valid_data = pd.read_csv("data/protein_valid.data", sep=" ", header=None)

clf = ensemble.BaggingClassifier()

model_CV = model_selection.GridSearchCV(clf, n_jobs=-1, cv=10)
model_CV.fit(X_data, y_data);

best_model = model_CV.best_estimator_

print(model_CV)
print("Best score is : ", model_CV.best_score_)