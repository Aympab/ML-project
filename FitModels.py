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


    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,random_state=0, test_size=0.2)
    scaler = RobustScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pca = PCA(n_components = 0.95)
    pca.fit(X_train_scaled)
    X_train_scaled = pca.transform(X_train_scaled)
    X_test_scaled = pca.transform(X_test_scaled)

    # XGBoost
    xgclas = xgb.XGBClassifier(learning_rate=0.1, max_depth=10, n_estimators=250, random_state = 0)
    xgclas.fit(X_train_scaled, y_train)
    y_pred1 = xgclas.predict(X_test_scaled)
    print('XGBoost Classifier :accuracy_score', accuracy_score(y_test,y_pred1))

    output = open('ResultsAccuracyEduardo.txt', 'w')
    output.write(f'XGBoost Classifier :accuracy_score: {accuracy_score(y_test,y_pred1)}')


if __name__ == '__main__':
    main(-1)
