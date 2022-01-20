from utils import *
from sklearn import cluster
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import KernelPCA

X, y, X_test, X_valid = load_data("data")

# Xtr, Xte, ytr, yte = model_selection.train_test_split(X, y, 
#                                                       test_size=0.2, 
#                                                       random_state=0)
print("Data loaded")

pipe = Pipeline([('scaler', StandardScaler()),
                ('reduction', KernelPCA()),
                ('simple_tree', KNeighborsClassifier(algorithm='brute',
                                                        leaf_size=50,
                                                        n_neighbors=10,
                                                        weights='uniform'))
                ])

grid_model = model_selection.GridSearchCV(pipe,
                            param_grid={},
                            #param_grid = {'reduction__n_components':[0.5, 0.6]},
                            param_grid = {'reduction__n_components':[250, 400, 500, 600, 750],
                                          'reduction__kernel':['poly', 'cosine'],
                                          'reduction__eigen_solver' : ['randomized']
                                          },
                            scoring = 'balanced_accuracy',
                            cv = 5,
                            verbose=3,
                            n_jobs=-1)


print("Grid search OK !")

grid_model.fit(X, y)

print(grid_model.best_params_)

y_test = grid_model.predict(X_test)
y_valid = grid_model.predict(X_valid)

np.savetxt("protein_test.predict", y_test, fmt="%d")
np.savetxt("protein_valid.predict", y_valid, fmt="%d")
zip_obj = ZipFile('submission_pipeline.zip', 'w')
zip_obj.write("protein_test.predict")
zip_obj.write("protein_valid.predict")

print("DONE")
