from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def LinReg(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    clf = LinearRegression()
    clf.fit(X_train, Y_train)
    return accuracy_score(Y_test, clf.predict(X_test).round())
