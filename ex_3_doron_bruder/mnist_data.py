import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models import Logistic, DecisionTree, KNearestNeighbor, SVM
import time

def load_data(path):
    data = np.loadtxt(path,delimiter=",")
    x,y = data[:,1:],data[:,0]
    valid_lables = np.logical_or((y==0),(y==1))
    return x[valid_lables],y[valid_lables]

def rearrange_data(X):
    return np.array([ x.flatten() for x in X ])

def compare_performance(X_train, y_train, x_test, y_test):
    M = [50, 100, 300, 500]
    models_num = 4

    def iteration(m,X_test,y_test,X_train,y_train):
        models = [Logistic(), DecisionTree(), KNearestNeighbor(), SVM()]
        X_train_m, ignore1, y_train_m, ignore2 = train_test_split(X_train, y_train, train_size=m, test_size=0.25)
        while (0 not in y_train_m) or (1 not in y_train_m):
            X_train_m, ignore1, y_train_m, ignore2 = train_test_split(X_train, y_train, train_size=m,
                                                                      test_size=0.25)
        ret = []
        for model in models:
            start_time = time.time()
            model.model.fit(X_train_m,y_train_m)
            elapsed_time = time.time() - start_time
            print("train : {} takes {}".format(model, elapsed_time))
            ret.append(model.model.score(X_test, y_test))
        return np.array(ret)

    def mean_performance(X_test, y_test, X_train, y_train):
        ret = []
        for m in M:
            mean = np.zeros(models_num)
            for i in range(50):
                mean += iteration(m, X_test, y_test, X_train, y_train)
            ret.append(mean/50)
        return np.array( ret )

    mean_performance = mean_performance(x_test,y_test,X_train,y_train)
    for _model_num, _name  in enumerate(["Logistic", "DecisionTree", "KNearestNeighbor", "soft-SVM"]):
        plt.plot( M , mean_performance[:,_model_num] )
    plt.legend( ["Logistic", "DecisionTree", "KNearestNeighbor", "soft-SVM"] )
    plt.title("calc_mean_performance")
    plt.xlabel("m (size of the given training data)")
    plt.ylabel("propability of successes")
    plt.show()


if __name__ == '__main__':
    x_train, y_train = load_data('mnist_train.csv')
    x_test, y_test = load_data('mnist_test.csv')
    compare_performance(rearrange_data(x_train), rearrange_data(y_train), rearrange_data(x_test), rearrange_data(y_test))