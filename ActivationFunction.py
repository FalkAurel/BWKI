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
    #S.214 und Analysis 1 Königsberger nachlesen
    def forward(self, inputs):
        """
        f(x) = max(0, x)
        """
        self.input = inputs
        return np.maximum(0, inputs)
    
    def backward(self, gradient):
        """
        Ableitung von max(x,y) -> 1(x > y). Oder im Falle der ReLU 1(x > 0). Sprich alle Werte größer werden mit dem Fak-
        tor 1 multipliziert; werden unverändert übernommen. Alle Werte kleiner gleich 0 werden mit dem Faktor 0 multipliz-
        iert; müssen einfach 0 gesetzt werden.
        """
        dInputs = gradient.copy()
        dInputs[self.input <= 0] = 0
        return dInputs
    
class Softmax(ActivationFunction):
    
    def forward(self, inputs):
        """
        S(i,j) = e^i,j / sum(e^i,j)
        Ausgabe ist ein 2D-Vektor  von Wahrscheinlichkeiten , die zu 1 (-> 100%) summieren.
        Edit: Beim Testen haben wir OverflowError bekommen schon bei np.exp(1000). Lösung zu diesem Problem ist dafür zu
        sorgen, dass der größte Wert 0 ist da np.exp(0) = 1 ist und alle anderen Werte sich zwischen 0 = np.exp(-np.inf)
        und 1 abspielen.
        """
        if inputs.ndim == 1:
            inputs = np.ones((1, inputs.shape[0])) * inputs
        expo = np.exp(inputs - np.max(inputs, axis = -1, keepdims = True))
        norm = np.sum(expo, axis = 1, keepdims = True)
        self.output = expo / norm
        return self.output
    
    def backward(self, gradient):
        """
        Noch nicht optimiert
        """
        dInputs = np.empty_like(gradient)
        for index, (output, derivative) in enumerate(zip(self.output, gradient)):
            output = output.reshape(-1, 1)
            jacobianMatrix = np.diagflat(output) - np.dot(output, output.T)
            dInputs[index] = np.dot(jacobianMatrix, derivative)
        return dInputs