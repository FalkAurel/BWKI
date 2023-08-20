import numpy as np
from abc import ABC, abstractmethod
    
class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, inputs):
        pass
    
    @abstractmethod
    def backward(self, gradient):
        pass
    
class ReLU(ActivationFunction):
    def forward(self, inputs):
        self._input = inputs
        return np.maximum(0, inputs)
    
    def backward(self, gradient):
        dInputs = gradient.copy()
        dInputs[self._input <= 0] = 0
        return dInputs

class LReLU(ActivationFunction):
    def __init__(self, alpha = 1e-3):
        self.alpha = alpha
        
    def forward(self, inputs):
        self._input = inputs
        return np.maximum(inputs * self.alpha, inputs)
    
    def backward(self, gradient):
        return np.where(self._input < 0, self.alpha, 1) * gradient

class ALReLU(ActivationFunction):
    #https://arxiv.org/abs/2012.07564
    def __init__(self, alpha=1e-3):
        self.alpha = alpha
    
    def forward(self, inputs):
        self._input = inputs
        return np.maximum(inputs * -self.alpha, inputs)
    
    def backward(self, gradient):
        return np.where(self._input < 0, -self.alpha, 1) * gradient

class tanh(ActivationFunction):
    def forward(self, inputs):
        self._input = np.tanh(inputs)
        return self._inputs
    
    def backward(self, gradient):
        return (1 - self._input**2) * gradient
    
class Softmax(ActivationFunction):
    #Quelle: https://www.pinecone.io/learn/softmax-activation/
    """
    Implementation der Softmaxaktivierungsfunktion. Eignet sich gut für Non-binary Classifier.
    """
    def forward(self, inputs):
        """
        input: ndarray -> output: ndarry of shape input.shape
        Gibt eine Wahrscheinlichkeitsverteilung zurück. Alle Werte eines samples sind in der
        Summe 1. 
        """
        expo = np.exp(inputs - np.max(inputs, axis = -1, keepdims = True))
        self._output = expo / np.sum(expo, axis = 1, keepdims = True)
        return self._output
    
    def backward(self, gradient):
        """
        input: ndarray of shape self._output.shape -> output: ndarray of shape input.shape
        Berechnet die derivatives der Werte, die in die Forwardmethode gegeben worden sind. 
        """
        dInputs = np.empty_like(gradient)
        for index, (output, derivative) in enumerate(zip(self._output, gradient)):
            output = output.reshape(-1, 1)
            jacobianMatrix = np.diagflat(output) - np.dot(output, output.T)
            dInputs[index] = np.dot(jacobianMatrix, derivative)
        return dInputs

class Sigmoid(ActivationFunction):
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output
    
    def backward(self, gradient):
        return gradient * (1 - self.output) * self.output
        return self.output
    
    def backward(self, gradient):
        return gradient * (1 - self.output) * self.output
