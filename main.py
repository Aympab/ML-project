import numpy as np
import xgboost as xgb
from sklearn.decomposition import KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import FeatureAgglomeration
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from utils import *

print("###BEGIN###")
print("Loading data...")
X, y, X_test, X_valid = load_data("data")

################################################################################
##############################  SCALING  #######################################
################################################################################
print("Scaling...")
# scaler = StandardScaler()
scaler = RobustScaler()
# scaler = QuantileTransformer(n_quantiles=500)
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)

################################################################################
##############################  REDUCTION  #####################################
################################################################################
print("Reduction...")
transformer = FeatureAgglomeration(n_clusters=650)
# transformer = GaussianRandomProjection(n_components=800)
# transformer = KernelPCA(kernel='poly' ,n_components=423)
X = transformer.fit_transform(X)
X_test = transformer.transform(X_test)
X_valid = transformer.transform(X_valid)

################################################################################
##############################  MODELS  ########################################
################################################################################
mlp = MLPClassifier(activation='logistic', learning_rate='adaptive', alpha=0.01)

model_dict = {
          'MLP'    : mlp,
          
        #   'RForest' : RandomForestClassifier(max_depth=50,
        #                                      n_estimators=200,
        #                                      max_features=0.15,
        #                                      n_jobs=-1),
          
          'KNN'     : KNeighborsClassifier(algorithm='kd_tree',
                                           leaf_size=50,
                                           n_neighbors=5,
                                           p=1,
                                           weights='distance',
                                           n_jobs=-1),
          
          'XGBoost' : xgb.XGBClassifier(learning_rate=0.1,
                                        max_depth=10,
                                        n_estimators=300,
                                        n_jobs=-1),
          
          'SelfTrainC' : SelfTrainingClassifier(mlp)
          }

################################################################################
###########################  PREDICTIONS  ######################################
################################################################################
for m_name, m_model in model_dict.items():
    print("MODEL : ", m_name)
    
    try:
        print("   >>> Fitting...")
        m_model.fit(X, np.ravel(y))

        print("   >>> Submitting...")
        submit_model(m_model, X_test, X_valid, name=m_name)

        scores = cross_val_score(m_model, X, np.ravel(y),
                        n_jobs=-1,
                        scoring='balanced_accuracy',
                        cv=3)
        
        print("   >>> Score " + m_name + ":", np.mean(scores))
        
    except Exception as e:
        print(" /!\ An error occured : ", e)
        continue
    
    print("   >>> Model submitted !")
    
print("###END###")

