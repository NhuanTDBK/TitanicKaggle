import pandas as pd
from pandas import Series,DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from __init__ import *

X_train, Y_train = load_file()
param_grid = {
	"C":[0.1,1,10,100,100],
	"kernel":["linear","rbf"]
}
clf = SVC(C=1, kernel='linear')

grid_search = GridSearchCV(clf, param_grid=param_grid)
grid_search.fit(X_train,Y_train)
svc = grid_search.best_estimator_
result = []
for idx, (train,test) in enumerate(kfold):
    svc.fit(X_train[train], Y_train[train])    
    Y_pred = svc.predict(X_train[test])
    result.append(svc.score(X_train[test], Y_train[test]))
print "Mean score : %s"%(np.array(score).mean())
