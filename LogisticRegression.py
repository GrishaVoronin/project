from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def LogReg(X, Y, classes):
    if classes == 3:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        X0_train = X_train[Y_train == 0]
        X1_train = X_train[Y_train == 1]
        X2_train = X_train[Y_train == 2]

        Y0_train = Y_train[Y_train == 0]
        Y1_train = Y_train[Y_train == 1]
        Y2_train = Y_train[Y_train == 2]

        clf1 = LogisticRegression(solver='saga')
        clf2 = LogisticRegression(solver='saga')
        clf3 = LogisticRegression(solver='saga')

        clf1.fit(np.concatenate((X0_train, X1_train), axis=0), np.concatenate((Y0_train, Y1_train), axis=0))
        pred1 = clf1.predict(X_test)

        clf2.fit(np.concatenate((X1_train, X2_train), axis=0), np.concatenate((Y1_train, Y2_train), axis=0))
        pred2 = clf2.predict(X_test)

        clf3.fit(np.concatenate((X0_train, X2_train), axis=0), np.concatenate((Y0_train, Y2_train), axis=0))
        pred3 = clf3.predict(X_test)

        pred = np.array([])
        for i in range(len(X_test)):
            pred = np.append(pred, np.argmax(np.bincount(np.array([pred1[i], pred2[i], pred3[i]]))))

        return accuracy_score(Y_test, pred)

    if classes == 2:
        clf = LogisticRegression(solver='saga')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        clf.fit(X_train, Y_train)
        pred = clf.predict(X_test)
        return accuracy_score(Y_test, pred)

    return 'Использование логистической регрессии при таком количестве классов - нецелесообразно'
