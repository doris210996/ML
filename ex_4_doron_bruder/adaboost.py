from ex4_tools import *
from matplotlib import pyplot as plt

class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.weak_learner = WL
        self.boosting_level = T
        self.boosted_learners = []
        self.weights = []

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        D = np.array([1.0/y.size]*y.size)
        self.weights = []
        normalize = lambda vec: vec / np.sum(vec ) if np.sum(vec ) != 0 else 0
        i = 0
        while True:
            if i == self.boosting_level:
                return D
            self.boosted_learners.append(self.weak_learner(D, X, y))
            pred = self.boosted_learners[i].predict(X)
            w = 0.5 * np.log( 1/np.sum( D[pred != y] ) - 1 )
            self.weights.append(w)
            D = normalize(D * np.e **( -w * y * pred )*10)
            i+=1

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        predictions = np.array([self.boosted_learners[i].predict(X) for i in range(max_t)])
        return np.sign(np.array(self.weights[:max_t])@predictions)

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the correct predictions when predict only with max_t weak learners (float)
        """
        return np.sum(np.ones(y.size)[self.predict(X, max_t) != y ])/y.size


def main():
    for noise in [0, 0.01, 0.4]:
        A = AdaBoost(DecisionStump, 500)
        X, y = generate_data(5000, noise)
        A.train(X, y)
        plt.plot(np.arange(500), [A.error(X, y, t) for t in range(500)])
        X, y = generate_data(200, noise)
        plt.plot(np.arange(500), [A.error(X, y, t) for t in range(500)])
        plt.legend(["Train err rate", "Test err rate"])
        plt.xlabel(" # Learners")
        plt.ylabel("Error")
        plt.ylim([0, 1])
        plt.title(f"Error rate vs nosie - {noise}")
        plt.show()
        W = np.array(A.weights).T
        W = (30 * W / max(W))[:y.size]
        for i, T in enumerate([5, 10, 50, 100, 200, 500]):
            plt.subplot(320 + i + 1)
            decision_boundaries(A, X, y, num_classifiers=T, weights=W)
        plt.show()


if __name__ == "__main__":
    main()
