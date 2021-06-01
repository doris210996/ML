from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from numpy import linalg
from utils import *

class Classfier:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        self.model.fit(self.bias(X), y)

    def predict(self, X):
        return self.model.predict(self.bias(X))

    def score(self,X,y):
        num_samples = len(X)
        miss , correct = 0,0
        FP,TP,TN = 0,0,0
        N , P = 0,0
        res_vec = self.predict(X)
        for i in range(num_samples):
            if y[i] == 1:
                P += 1
            else:
                N += 1
            if res_vec[i] == 1 and y[i] == -1*1:
                TN += 1
            if res_vec[i] == 1 and y[i] == 1:
                TP += 1
            if res_vec[i] == 1 and y[i] == -1*1:
                FP += 1
            if res_vec[i] == y[i]:
                correct += 1
            else:
                miss += 1
        FPR = float(FP)/N
        TPR = float(TP)/P
        precision = float(TP)/TP+FP
        specificty = float(TN)/N
        error = float(miss)/num_samples
        accuracy = float(correct)/num_samples
        return {'FPR':FPR,'TPR':TPR,'precision':precision,'specificty':specificty,'error':error,'accuracy':accuracy}

    def bias(self, X):
        return np.insert(X,0, 1, 1)


class HalfSpace(Classfier):

    def __init__(self):
        super().__init__()

    def fit(self,X,y):
        def _perceptron(X, y):
            w = np.zeros(shape=X.shape[1])
            while True:
                for i in range(len(y)):
                    if y[i] * (np.dot(w, X[i])) <= 0:
                        w = w + (y[i]*X[i])
                else:
                    return w
        self.model = _perceptron(self.bias(X),y)

    def predict(self,U):
        U = self.bias(U)
        return np.sign(self.model @ U.T)

class LDA(Classfier):
    def __init__(self):
        super().__init__()
    ## Notes:
    ## As we dont know D s.t x | y ~ D so we will use estimatiors
    ## The estimator of x|y are:
    ## 1. For calc the est-miu vector we will take avg of the relevant samples
    ## 2. For calc the est-cov matrix we use np.cov which get the cov est of cols ,so we use transpose the use
    ## the rows as samples
    def fit (self, X, y):
        def calc_for_deltas(X,y_val):
            y_val_rows = self.bias(X)[y == y_val]
            ln_py,inv_sigma = np.log(y_val_rows.size / y.size),np.linalg.pinv(np.cov(y_val_rows.T))
            miu = np.mean(y_val_rows,axis=0)
            return miu,ln_py,inv_sigma
        delta_y = lambda x,inv_sigma,miu,ln_p: x.T @ inv_sigma @ miu - 0.5 * miu.T @ inv_sigma @ miu + ln_p
        miu1,ln_py1,inv_sigma1 = calc_for_deltas(X,1)
        miu2, ln_py2, inv_sigma2 = calc_for_deltas(X,-1)
        arg_max = lambda x: 1. if delta_y(x,inv_sigma1,miu1,ln_py1)  >= delta_y(x,inv_sigma2,miu2,ln_py2) else -1.
        self.model = arg_max

    def predict(self, U):
        return np.apply_along_axis(self.model,1,self.bias(U))

class SVM(Classfier):
    def __init__(self):
        super().__init__()
        self.model = SVC(C=1e10, kernel='linear')
    def fit(self, X, y):
        self.model.fit(self.bias(X), y)

    def predict(self, X):
        return self.model.predict(self.bias(X))

class Logistic(Classfier):
    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(solver='liblinear')

class DecisionTree(Classfier):
    def __init__(self):
        super().__init__()
        self.model = DecisionTreeClassifier()

class KNearestNeighbor(Classfier):
    def __init__ (self):
        super().__init__()
        self.model =  KNeighborsClassifier(n_neighbors=40)

def draw_points(m):
    label = lambda x: np.sign(np.dot([0.3, -0.5], x) + 0.1)
    X = np.random.multivariate_normal(np.zeros(2), np.identity(2), m)
    y = np.apply_along_axis(label,1,X)
    return X,y

def compare_performance():
        def iteration(X,y,svm,lda,prec):
            prec = HalfSpace()
            prec.fit(X,y)
            lda.fit(X,y)
            clf = svm.model
            clf.fit(X, y)
            w = clf.coef_[0]
            a = -w[0] / w[1]
            xx = np.linspace(-5, 5)
            yy = a * xx - (clf.intercept_[0]) / w[1]
            def get_y(W, _x):
                return -(W[0] + _x * W[1]) / W[2] if W[2] != 0 else -W[0]
            ax = plt.axes()
            get_prec_y = lambda x: get_y(prec.model,x)
            f_x_prec = np.apply_along_axis(get_prec_y,0,xx)
            ax.plot(xx, f_x_prec,label='prec')
            plt.legend()
            f_truth = lambda x: get_y([0.1,0.3,-0.5],x)
            f_x_truth = np.apply_along_axis(f_truth,0,xx)
            ax.plot(xx, f_x_truth,label='truth')
            plt.legend()
            plt.plot(xx, yy, 'k-',label='svm')
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
            plt.legend()
            plt.axis('tight')
            plt.show()
            svd_arr, lda_arr, precp_arr = [], [], []
            for i in range(500):
                X_test, y_test = draw_points(10000)
                while (-1 not in y_test) or (1 not in y_test):
                    X_test, y_test = draw_points(10000)
                precp_arr.append(prec.score(X_test,y_test)['accuracy'])
                svd_arr.append(clf.score(X_test,y_test))
                lda_arr.append(lda.score(X_test,y_test)['accuracy'])
            return np.mean(precp_arr),np.mean(svd_arr),np.mean(lda_arr)
        svm_scores,prec_scores,lda_scores= [],[],[]
        for m in [5, 10, 15, 25, 70]:
            X_train, y_train = draw_points(m)
            svm,precp,lda = SVM(),HalfSpace(),LDA()
            prec_score,svm_score,lda_score = iteration(X_train,y_train,svm,precp,lda)
            svm_scores.append(svm_score)
            prec_scores.append(prec_score)
            lda_scores.append(lda_score)
        x = [5, 10, 15, 25, 70]
        y1 = np.array(prec_scores)
        y2 = np.array(svm_scores)
        y3 = np.array(lda_scores)
        plt.plot(x, y1, label="prec")
        plt.plot(x, y2, label="svm")
        plt.plot(x, y3, label="lda")
        plt.xlabel('m (size of the given training data)')
        plt.ylabel('probability of successes')
        plt.title('calc_mean_performance')
        plt.legend()
        plt.show()






if __name__ == "__main__" :
    compare_performance()




