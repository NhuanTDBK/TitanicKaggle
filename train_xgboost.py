import xgboost as xgb
import pandas as pd
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import StratifiedKFold
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import numpy as np
import pickle

n_iter = 2000
k_fold = 10

file_dat = np.load("train_test_v2.npz")
X_train = file_dat["X_train"]
Y_train = file_dat["y_train"]
X_test = file_dat["X_test"]  
cv = StratifiedKFold(Y_train,n_folds=k_fold,shuffle=True)
# initialize the classifier
GB = xgb.XGBClassifier()
  
param_grid = {'max_depth': sp_randint(1, 100),
              'learning_rate': sp_uniform(loc=0e0,scale=1e0),
              'objective':['binary:logistic'],
              'nthread': [8],
              'missing': [np.nan],
              'reg_alpha': [0.01,0.017782794,0.031622777,0.056234133,\
                            0.1,0.17782794,0.31622777,0.56234133,1.,1.77827941,\
                            3.16227766,5.62341325,10.,\
                            17.7827941,31.6227766,56.2341325,100.],
              'colsample_bytree': sp_uniform(loc=0.2e0,scale=0.8e0),
              'subsample': np.arange(0.6,1.0,step=0.05),
              'n_estimators': sp_randint(200,800)
	}
  
search_GB = RandomizedSearchCV(GB,param_grid,scoring='accuracy',\
               n_iter=n_iter,cv=cv,verbose=True)
search_GB.fit(X_train,Y_train)
print ' ',search_GB.best_score_
print ' ',search_GB.best_params_
  
# save the results
#f_name = open('xgboost_RSCV_%s.dat'%search_GB.best_score_,'w')
#pickle.dump(search_GB,f_name)
#f_name.close()
pd.DataFrame(search_GB.best_estimator_.predict(X_test)).to_csv('result_%s.csv'%search_GB.best_score_)
