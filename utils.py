from tabnanny import verbose
import numpy as np
import pandas as pd
from zipfile import ZipFile
from sklearn import preprocessing, decomposition

def load_data(path="/data"):
    X = pd.read_csv(path + "/protein_train.data", sep=" ", header=None)
    y = pd.read_csv(path + "/protein_train.solution", sep=" ", header=None)

    X_test = pd.read_csv(path + "/protein_test.data", sep=" ", header=None)
    X_valid = pd.read_csv(path + "/protein_valid.data", sep=" ", header=None)

    return X, y, X_test, X_valid

def submit_model(regressor, X_test, X_valid):
    
    y_test = regressor.predict(X_test)
    y_valid = regressor.predict(X_valid)
    
    np.savetxt("protein_test.predict", y_test, fmt="%d")
    np.savetxt("protein_valid.predict", y_valid, fmt="%d")
    zip_obj = ZipFile('submission', 'w')
    zip_obj.write("protein_test.predict")
    zip_obj.write("protein_valid.predict")
    
    return