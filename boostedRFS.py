import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def boostedRFS(X, Y):
    clf = xgb.XGBClassifier()
    params = {'n_estimators': np.arange(1, 42, 5), 'max_depth': np.arange(1, 8)}
    clf_grid = GridSearchCV(clf, params, cv=5, scoring='accuracy', n_jobs=-1, refit=1)
    clf_grid.fit(X, Y)
    print(clf_grid.best_params_)
    pred = clf_grid.predict(X)
    return accuracy_score(Y, pred)
