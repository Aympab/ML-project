import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import FeatureAgglomeration
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from utils import *

print("###BEGIN###")
print("Loading data...")
X, y, X_test, X_valid = load_data("data")

################################################################################
##############################  SCALING  #######################################
################################################################################
print("Scaling...")
scaler = RobustScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)

################################################################################
##############################  REDUCTION  #####################################
################################################################################
print("Reduction...")
transformer = FeatureAgglomeration(n_clusters=600)
X = transformer.fit_transform(X)
X_test = transformer.transform(X_test)
X_valid = transformer.transform(X_valid)

################################################################################
##############################  MODELS  ########################################
################################################################################
model_dict = {'MLP1'    : MLPClassifier(activation='tanh',
                                    learning_rate='adaptive'),
          
          'MLP2'    : MLPClassifier(activation='logistic',
                                    learning_rate='adaptive',
                                    alpha=0.01),
          
          'RForest1' : RandomForestClassifier(max_depth=50,
                                             n_estimators=200,
                                             max_features=0.15),
          
          'RForest2' : RandomForestClassifier(max_depth=50,
                                             n_estimators=200),
          
          'KNN'     : KNeighborsClassifier(algorithm='kd_tree',
                                           leaf_size=50,
                                           n_neighbors=5,
                                           p=1,
                                           weights='distance'),
          
          'XGBoost' : xgb.XGBClassifier(learning_rate=0.1,
                                        max_depth=10,
                                        n_estimators=300),
          
          'SelfTrainC' : SelfTrainingClassifier(
                                    MLPClassifier(activation='tanh',
                                                  learning_rate='adaptive')
                                    )
          }

################################################################################
###########################  PREDICTIONS  ######################################
################################################################################
for m_name, m_model in model_dict.items():
    print("MODEL : ", m_name)
    
    try:
        print(" >>> Fitting...")
        m_model.fit(X, np.ravel(y))

        print(" >>> Submitting...")
        submit_model(m_model, X_test, X_valid, name=m_name)

    except Exception as e:
        print(" /!\ An error occured : ", e)
        continue
    
    print(">>> Model submitted !")
    
print("###END###")

