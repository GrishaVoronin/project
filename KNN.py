from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def KNN(X, Y):
    clf = KNeighborsClassifier()
    # Выявлление лучших параметров с помощью GridSearchCV
    params = {'n_neighbors': [i for i in range(1, 11)],
              'weights': ['uniform', 'distance'],
              'metric': ['manhattan', 'euclidean']}
    clf_grid = GridSearchCV(clf, params, cv=5, scoring='accuracy', n_jobs=-1, refit=1)
    clf_grid.fit(X, Y)
    print(clf_grid.best_params_)
    return accuracy_score(Y, clf_grid.predict(X))
