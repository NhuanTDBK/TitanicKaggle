import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
kfold = StratifiedKFold(Y_train, n_folds=10)
def load_file():
	dat = np.load("train_test.npz")
	X_train = dat["X_train"]
	Y_train = dat["y_train"]
	return X_train, Y_train
