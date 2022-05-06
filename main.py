from sklearn.datasets import load_wine, load_iris, load_breast_cancer

from KNN import KNN
from LogisticRegression import LogReg
from LinReg import LinReg
from tree import tree
from RFC import RFS
from boostedRFS import boostedRFS

models = [LinReg, LogReg, KNN, tree, RFS, boostedRFS]
models_names = ['Линейная регрессия', 'Логистическая регрессия', 'Метод k-ближайших соседей', 'Решающее дерево', 'Случайный лес', 'Случайный лес + градиентный бустинг']
datasets = [load_iris, load_wine, load_breast_cancer]
datasets_names = ['Ирисы Фишера', 'Распознование вин', 'Рак груди']

def start():
    for i in range(3):
        print(datasets_names[i])
        print('')
        dataset = datasets[i]()
        X, Y = dataset['data'], dataset['target']
        for j in range(6):
            print(models_names[j])
            if j == 1:
                if i == 2:
                    accuracy = models[j](X, Y, 2)
                else:
                    accuracy = models[j](X, Y, 3)
            else:
                accuracy = models[j](X, Y)
            print(f'Точность: {accuracy}')
            print('')
        print('')
start()

