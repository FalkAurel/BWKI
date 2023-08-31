import numpy as np

class logisticRegression():
    def fit(self, X, y, *, epoch = 1000, learningRate = 1e-3):
        """
        X is expected to be 2d array, even for simple logistic Regression. y is expected to be a 1d array of all targets.
        Using the maximum likelihood approach we estimate the coefficents. Updating is done via gradient descent.
        The derivative with respect to the the the coefficient is
        b_i = sum((p - y) * X_i)
        """
        X, y = np.hstack((np.ones((len(X), 1)), X)), y.reshape(-1, 1)
        self.coefficients = np.random.randn(X.shape[-1]).reshape(-1, 1)
        for _ in range(epoch):
            p = self.sigmoid(X @ self.coefficients)
            gradient = X.T @ (p - y)
            self.coefficients -= learningRate * gradient
        return self
    
    def evaluate(self, X, y, *, decisionBoundry = 0.5, predictions = False):
        """
        X is expected to be 2d array, even for simple logistic Regression. y is expected to be a 1d array of all targets.
        In case you intend to only use the predicted values you can pass anything for y, as it'll be not used in this case.
        """
        X = np.hstack((np.ones((len(X), 1)), X))
        yhat = self.sigmoid(X @ self.coefficients)
        if predictions:
            return yhat
        return np.mean(np.where(yhat >= decisionBoundry, 1, 0).ravel() == y)
    
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
    def __repr__(self):
        return f"Coefficient= {self.coefficients}"
