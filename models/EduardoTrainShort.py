import pandas as pd
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern


from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

def main(njobs):
    X_data = pd.read_csv("data/protein_train.data", sep=" ", header=None)
    y_data = pd.read_csv("data/protein_train.solution", sep=" ", header=None)

    X_test_data = pd.read_csv("data/protein_test.data", sep=" ", header=None)
    X_valid_data = pd.read_csv("data/protein_valid.data", sep=" ", header=None)
    output = open('GridSerachResultsEduardoShort.txt', 'w')

    X_data2 = X_data.iloc[:15].copy()
    y_data2 = y_data.iloc[:15].copy()

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,random_state=0, test_size=0.2)
    scaler = RobustScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pca = PCA(n_components = 0.95)
    pca.fit(X_train_scaled)
    X_train_scaled = pca.transform(X_train_scaled)
    X_test_scaled = pca.transform(X_test_scaled)

    # XGboost
    tuned_parameters = [{'max_depth': [5, 10, 20, 30],'learning_rate':[0.01, 0.1, 0.5], 'n_estimators': [100, 150, 250]}]
    clas = GridSearchCV(xgb.XGBClassifier(), tuned_parameters, cv=3, n_jobs=-njobs)
    clas.fit(X_train_scaled, y_train)
    print('The best hyper-parameters for XGBBoost are: ',clas.best_params_)
    output.write(f'The best hyper-parameters for XGBBoost are: {clas.best_params_}\n')

    # Random Forests
    tuned_parameters = [{'max_depth': [5,10, 15, 20, 50, 70], 'n_estimators': [10, 25, 50, 100,150, 200, 250]}]
    clas_rf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=3, n_jobs=njobs)
    clas_rf.fit(X_train_scaled, y_train)
    print('The best hyper-parameters for Random Forests are: ',clas_rf.best_params_)
    output.write(f'The best hyper-parameters for Random Forests are: {clas_rf.best_params_}\n')

    #KNN
    tuned_parameters = [{'n_neighbors': [1,2,3,4,5,10,15,20], 'p': [1,2]}]
    model = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=4, n_jobs=njobs)
    model.fit(X_train_scaled, y_train)
    print('The best hyper-parameters for KNN are: ', model.best_params_)
    output.write(f'The best hyper-parameters for KNN are: {model.best_params_}\n')


    # SVM
    tuned_parameters = [{'kernel': ['linear', 'rbf', 'poly'], 'C':[1, 2, 3, 5, 6, 7, 10], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1]}]
    svm_clas = GridSearchCV(SVC(), tuned_parameters, cv=3, n_jobs=njobs)
    svm_clas.fit(X_train_scaled, y_train)
    print('The best hyper-parameters for SVR are: ', svm_clas.best_params_)
    output.write(f'The best hyper-parameters for SVR are: {svm_clas.best_params_}\n')


    # Decision Tree
    tuned_parameters = [{'max_depth': [5,10, 15, 20, 50, 100,200]}]
    clas_dt = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=3, n_jobs=njobs)
    clas_dt.fit(X_train_scaled, y_train)
    print('The optimum max_depth for Decision Tree is: ', clas_dt.best_params_ )
    output.write(f'The optimum max_depth for Decision Tree is: {clas_dt.best_params_ }\n')


    # GP
    kernel1 = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    kernel2 = C() + Matern(length_scale=0.5, nu=3/2)
    kernel3 = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) +  C(1.0, (1e-3, 1e3))* Matern(length_scale=2, nu=3/2)
    kernel4 = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))

    tuned_parameters = [{'kernel': [kernel1,kernel2,kernel3,kernel4],
                       'n_restarts_optimizer':[5]}]
    gpclas = GridSearchCV(GaussianProcessClassifier(), tuned_parameters, cv=3, n_jobs=njobs)
    gpclas.fit(X_train_scaled, y_train)
    print('The best hyper-parameters for GPR are: ', gpclas.best_params_ )
    output.write(f'The best hyper-parameters for GPR are: {gpclas.best_params_}\n')

    output.close()

if __name__ == '__main__':
    main(-1)
