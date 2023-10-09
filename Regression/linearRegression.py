import numpy as np
import matplotlib.pyplot as plt

class simpleLinearRegression():
    def fit(self, X, y):
        """
        Solving for the beta0 and beta1 coefficients. beta1 is calculated using the formula:
        beta1 = sum((x_i - x̅)(y_i - ȳ)) / sum((x_i - x̅)^2)
        where x̅ and ȳ are the column means of X and Y, respectively.
        Once we have calculated beta1, we can solve for beta0 using the formula:
        beta0 = ȳ - beta1 * x̅
        This method calculates these coefficients for each column of the input data, resulting in a
        set of coefficients for each column.
        """
        xMean , yMean = X.mean(axis = 0), y.mean(axis = 0)
        self.beta1 = np.sum((X - xMean) * (y- yMean), axis = 0) / np.sum(np.square(X - xMean), axis = 0)
        self.beta0 = yMean - xMean * self.beta1
        return self
    
    def evaluate(self, X, y, *, tolerance = 0.05, visualize = False):
        """
        Evaluating the model. Returns by default a value between 0 and 1, which indicates how well the model has
        estimated the coefficient. The tolerance parameter is set to 0.05 by default(5%) which determines how
        harsh the evaluation is carried out. Furthermore it'll also return the RMSE (the lower, the better the fit).
        When setting visualize to true, the function will return all predicted values.
        """
        yhat = self.beta0 + self.beta1 * X
        if visualize:
            return yhat
        absoluteDifference, allowedDifference = np.abs(y - yhat), np.abs(y * (1 + tolerance) - y)
        return np.mean(absoluteDifference <= allowedDifference), np.sqrt(np.mean((y - yhat)**2))

    
    @property
    def getCoefficients(self):
        return self.beta0, self.beta1
    
    def visualizeCoefficients(self):
        """
        Each point describes the relationship between beta0 and beta1, as shown in the axes. The X-coordinate represents
        the intercept whereas the Y-coordinate represents the slope.
        """
        beta0, beta1 = self.getCoefficients
        plt.scatter(beta0, beta1)
        plt.xlabel("Beta0")
        plt.ylabel("Beta1")
        plt.title("Coefficients of Simple Linear Regression Model")
        plt.show()
        

class MultipleLinearRegression():
    def fit(self, X, y):
        """
        Using a very similar approach to PolynomnialRegression. We solve for the coefficients by applying matrix algebra.
        Which makes sense, since multiple lineare regression is the more general approach polynomnial regression. 
        """
        inputMatrix = np.hstack((np.ones((X.shape[0], 1)), X))
        self.coefficients = np.linalg.inv(inputMatrix.T @ inputMatrix) @ inputMatrix.T @ y
        return self
    
    def evaluate(self, X, y, *, tolerance = 0.05, visualize = False):
        """
        Evaluating the model. Returns by default a value between 0 and 1, which indicates how well the model has
        estimated the coefficient. The tolerance parameter is set to 0.05 by default(5%) which determines how
        harsh the evaluation is carried out. Furthermore it'll also return the RMSE (the lower, the better the fit).
        When setting visualize to true, the function will return all predicted values.
        """
        yhat = np.hstack((np.ones((X.shape[0], 1)), X)) @ self.coefficients
        if visualize:
            return yhat
        absoluteDifference, allowedDifference = np.abs(y - yhat), np.abs(y - y * (1 + tolerance))
        return np.mean(absoluteDifference <= allowedDifference), np.sqrt(np.mean((y - yhat) ** 2))
    
    def samplingData(self, X, y, *, threshold = 0.7):
        """
        using the coefficent of correlation we'll eliminate all dimensions in the data which don't show a high correlation
        with the target. The threshold parameter let's you finetune the elimination process.
        """
        manipulatedX, manipulatedY = X - X.mean(axis = -1), y -y.mean() 
        R = np.sum(manipulatedX * manipulatedY) / (np.sum(manipulatedX**2)**0.5 * np.sum(manipulatedY**2)**0.5)
        return np.delete(X, np.where(R > threshold)[0], axis=-1)
    
    @property
    def getCoefficients(self):
        return self.coefficients
    
    def visualizeCoefficients(self):
        """
        Each bar describes the coefficient. The X-axis shows us which coefficients it is and the y-Axis shows the value
        of the coefficient.
        """
        coefficients = self.coefficients.ravel()
        plt.bar(range(len(coefficients)), coefficients)
        plt.xlabel("Coefficient Index")
        plt.ylabel("Coefficient Value")
        plt.title("Coefficients of Multiple Linear Regression Model")
        plt.show()
