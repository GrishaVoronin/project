from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def RFS(X, Y):
    params = {'n_estimators': np.arange(1, 100, 5), 'max_depth': np.arange(1, 10)}
    clf = RandomForestClassifier()
    boosted_model_grid = GridSearchCV(clf, params, cv=5, scoring='accuracy', n_jobs=-1, refit=1)
    boosted_model_grid.fit(X, Y)
    return accuracy_score(Y, boosted_model_grid.predict(X))

