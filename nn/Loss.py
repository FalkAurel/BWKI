import numpy as np
from abc import ABC, abstractmethod

class LossFunction(object):
    
    def tranierbareLayer(self, layers):
        self.regularizationLayer = layers
    
    @abstractmethod
    def calculate(self, yhat, y):
        pass
    
    @abstractmethod
    def backward(self, yhat, y):
        pass
    
    def sumLoss(self, yhat, y):
        self.sum += (self.calculate(yhat, y))
        return self.sum
    
    def regularizationLoss(self):
        regularizationLoss = 0
        for layer in self.regularizationLayer:
            if layer.l1WFaktor:
                regularizationLoss += layer.l1WFaktor * np.abs(layer.weight).sum()
                regularizationLoss += layer.l1BFaktor * np.abs(layer.bias).sum()
            if layer.l2WFaktor:
                regularizationLoss += layer.l2WFaktor * np.sum(layer.weight**2)
                regularizationLoss += layer.l2BFaktor * np.sum(layer.bias**2)
        return regularizationLoss
    
class CategoricalCrossEntropyLoss(LossFunction):
    
    def calculate(self, yhat, y):
        if yhat.shape[0] == 1 and y.shape[0] == 1:
            yhat, y = yhat.reshape(-1), y.reshape(-1)
        return -np.mean(np.log(np.sum(np.clip(yhat, 1e-7, 1-1e-7) * y, axis= -1)))
    
    def backward(self, yhat, y):
        return -y/ (yhat * len(yhat))

class BinaryCrossEntropyLoss(LossFunction):
    """
    Implementation von BinaryCrossEntropyLoss. Anzuwenden bei Binary-Classification.
    """
    def calculate(self, yhat, y):
        yhat = np.clip(yhat, 1e-7, 1-1e-7)
        return -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
        
    def backward(self, yhat, y):
        yhat = np.clip(yhat, 1e-7, 1 - 1e-7)
        return -(y / yhat - (1 - y) / (1 - yhat)) / (len(yhat[0]) * len(yhat))
    
class MeanSquaredError(LossFunction):
    """
    Implementation von Mean squared Error(MSE). Für Regressionsprobleme anzuwenden.
    """
    def calculate(self, yhat, y):
        """
        f(yhat, y) = 1/nInputs * ∑(y - yhat)^2
        """
        return np.mean((y - yhat)**2)#, axis = -1)
    
    def backward(self, yhat, y):
        """
        Ableitung von f(yhat, y) nach yhat: 2(y - yhat) / nInput
        """
        return -2 * (y - yhat)/ (len(yhat) * len(yhat[0]))

class MeanAbsoluteError(LossFunction):
    """
    Implementation von Mean absolute Error(MAE). Für Regressionsprobleme anzuwenden.
    """
    def calculate(self, yhat, y):
        """
        f(yhat, y) = 1/nInput * ∑|y - yhat|
        """
        return np.mean(np.abs(y - yhat), axis = -1)
    
    def backward(self, yhat, y):
        """
        Ableitung von f(yhat, y) nach yhat: 1/nInput * {1 für y - yhat > 0; 0 für y - yhat < 0}
        """
        return np.sign(y - yhat) / (len(yhat) * len(yhat[0]))