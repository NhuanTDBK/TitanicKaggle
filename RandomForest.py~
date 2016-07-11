import pandas as pd
from pandas import Series,DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import StratifiedKFold
from __init__ import *


X_train, Y_train = load_file()
random_forest = RandomForestClassifier(n_estimators=100)
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
grid_search.fit(X_train,Y_train)
random_forest = grid_search.best_estimator_
result = []
for idx, (train,test) in enumerate(kfold):
    random_forest.fit(X_train[train], Y_train[train])    
    Y_pred = random_forest.predict(X_train[test])
    score = random_forest.score(X_train[test], Y_train[test])
    result.append(score)
    print score
print "Mean score : %s"%(np.array(score).mean())
