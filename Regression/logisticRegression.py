import numpy as np

class logisticRegression():
    def fit(self, X, y, *, epoch = 1000, lerningRate = 1e-3):
        self.beta0, self.beta1 = np.random.randn(1), np.random.randn(1)
        for _ in range(epoch):
            p = self.sigmoid(self.beta0 + self.beta1 * X)
            loggedLikelihood = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            Loss = -loggedLikelihood
            dbeta0 = np.sum(p - y)
            dbeta1 = np.sum((p - y) * X)
            self.beta0 -= lerningRate * dbeta0
            self.beta1 -= lerningRate * dbeta1
        return self
    
    def evaluate(self, X, y, *, decisionBoundry = 0.5, predictions = False):
        yhat = self.sigmoid(self.beta0 + self.beta1 * X)
        if predictions:
            return yhat
        return np.mean(np.where(yhat >= decisionBoundry, 1, 0) == y)
    
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
    def __repr__(self):
        return f"beta0 = {self.beta0}, beta1 = {self.beta1}"