from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np

def tree(X, Y):
    clf = DecisionTreeClassifier()
    params = {'max_depth': np.arange(1, 10)}
    clf_grid = GridSearchCV(clf, params, cv=5, scoring='accuracy', n_jobs=-1, refit=1)
    clf_grid.fit(X, Y)
    print(clf_grid.best_params_)
    return accuracy_score(Y, clf_grid.predict(X))
