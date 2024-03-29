import numpy as np
import matplotlib.pyplot as plt

class PolynomnialRegression():
    def __init__(self, order):
        self.order = order
    
    def fit(self, X, y):
        """
        Rewriting our Input in matrix-form. We'll recieve by doing so 1 inputMatrix and one coefficient Matrix.
        We can now solve for the coefficient Matrix.
        Approach: inputMatrix @ coefficientMatrix = y
                  inputMatrix.T @ inputMatrix @ coefficientMatrix = inputMatrix.T @ y
                  coefficientMatrix = (inputMatrix.T @ inputMatrix)^-1 inputMatrix @ y
        """
        X = X.reshape(-1, 1)
        inputMatrix = np.hstack([X**order for order in range(self.order + 1)])
        self.coefficients = np.linalg.inv(inputMatrix.T @ inputMatrix) @ inputMatrix.T @ y
        return self
        
    def evaluate(self, X, y, *, tolerance = 0.05, visualize = False):
        """
        Evaluating the model. Returns by default a value between 0 and 1, which indicates how well the model has
        estimated the coefficient. The tolerance parameter is set to 0.05 by default(5%) which determines how
        harsh the evaluation is carried out. Furthermore it'll also return the RMSE (the lower, the better the fit).
        When setting visualize to true, the function will return all predicted values.
        """
        X = X.reshape(-1, 1)
        inputMatrix = np.hstack([X**order for order in range(self.order + 1)])
        yhat = inputMatrix @ self.coefficients
        if visualize:
            return yhat
        absoluteDifference, allowedDifference = np.abs(y - yhat), np.abs(y - y * (1 + tolerance))
        return np.mean(absoluteDifference <= allowedDifference), np.sqrt(np.mean(y - yhat)**2)
    
    @property
    def getCoefficients(self):
        return self.coefficients
    
    def visualizeCoefficients(self):
        """
        Each bar describes the amount to which the corresponding coefficient has contributed to the output.
        The bigger the bar, the stronger the impact of the coefficient on the output.
        """
        coefficients = self.coefficients.ravel()
        plt.bar(range(len(coefficients)), coefficients)
        plt.xticks(range(len(self.coefficients)), [f"x^{i}" for i in range(len(coefficients))])
        plt.ylabel("Coefficient")
        plt.show()
    
    def __repr__(self):
        return f"PolynomnialRegression of function of order {self.order}"
