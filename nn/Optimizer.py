import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    
    @abstractmethod
    def __init__(self, lernrate, lernRateDecay):
        self.lernrate = lernrate
        self._stepCount = 0
        self._currentLearnRate = lernrate
        self._learnRateDecay = lernRateDecay
    
    @abstractmethod
    def step(self, layer):
        raise NotImplementedError
    
    def learningRateDecay(self):
        self._currentLearnRate = self.lernrate / (1. + self._stepCount * self._learnRateDecay)
        self._stepCount += 1
    
    @property
    def getLearningRate(self):
        return self._currentLearnRate
    
    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    
class SGD(Optimizer):
    """
    Stochastic Gradient Descent(SGD). Vanilla-Optimizer. Dieser hier supported learningRateDecay; kontrollierte Ab-
    nahme der Learning Rate und Momentum. Momentum ist echt interessant, da wir so aus einem lokalen Minimum(leicht-
    er / eher) ausbrechen können. Momentum ist nur das vorherrige Update zu dem Parameter mit einem Faktor zwischen
    0 - 1; wieviel davon übernommen werden soll.
    """
    
    def __init__(self, lernrate, lernRateDecay, momentum=0.9):
        super().__init__(lernrate, lernRateDecay)
        self._momentum = momentum
        
    def step(self, layer):
        if self._momentum:
            if not hasattr(layer, "weightMomentum"):
                layer.weightMomentum = np.zeros_like(layer.weight)
                layer.biasMomentum = np.zeros_like(layer.bias)
            updateWeight = self._momentum * layer.weightMomentum - self._currentLearnRate * layer.dweight
            updateBias = self._momentum * layer.biasMomentum - self._currentLearnRate * layer.dbias
            layer.weightMomentum = updateWeight
            layer.biasMomentum = updateBias
            layer.weight += updateWeight
            layer.bias += updateBias
        else:
            layer.weight -= self._currentLearnRate * layer.dweight
            layer.bias -= self._currentLearnRate * layer.dbias
    
    def __repr__(self):
        return f"SGD(LearningRate={self.lernrate}, learningRateDecay={self._learnRateDecay}, momentum={self._momentum})"


class AdaGrad(Optimizer):
    """
    AdaGrad ist eine "Subform" des SGD. AdaGrad steht für adaptive Gradient. Wie der Name vermuten lässt, adaptiert
    dieser. Dieses Adaption wird durch eine Normalisierung der Werte erreicht, indem man einen "Update-Verlauf" von
    dem Lernprozess anfertigt. Wenn dieser in eine Richtung zu stark ansteigt, so wird die Optimierung in diese Rich-
    tung immer langsamer.
    Der Update-Verlauf entsteht durch den ZwischenSpeicher, der jedes mal mit dem Quadrat der aktuellen Gradienten
    aktualisiert wird. Je größer dieser Wert wird, desto größer der Nenner.
    """
    
    def __init__(self, lernrate, lernRateDecay, epsilon = 1e-7):
        super().__init__(lernrate, lernRateDecay)
        self.epsilon = epsilon
    
    def step(self, layer):
        if not hasattr(layer, "weightZwischenSpeicher"):
            layer.weightZwischenSpeicher = np.zeros_like(layer.weight)
            layer.biasZwischenSpeicher = np.zeros_like(layer.bias)
        layer.weightZwischenSpeicher += layer.dweight**2
        layer.biasZwischenSpeicher += layer.dbias**2
        layer.weight -= self._currentLearnRate * layer.dweight / (np.sqrt(layer.weightZwischenSpeicher) + self.epsilon)
        layer.bias -= self._currentLearnRate * layer.dbias / (np.sqrt(layer.biasZwischenSpeicher) + self.epsilon)
    
    def __repr__(self):
        return f"AdaGrad(LearningRate={self.lernrate}, learningRateDecay={self._learnRateDecay}, epsilon={self.epsilon})"

class RMSProp(Optimizer):
    """
    Absolut Dogshit. Rechnung überprüfen. NOCH NICHT FERTIG MUSS NOCH OPTMIERT WERDEN!!!
    """
    def __init__(self, lernrate = 1e-3, lernRateDecay = 0, epsilon = 1e-7, rho = .9):
        super().__init__(lernrate, lernRateDecay)
        self.epsilon, self.rho = epsilon, rho
    
    def step(self, layer):
        if not hasattr(layer, "weightZwischenSpeicher"):
            layer.weightZwischenSpeicher = np.zeros_like(layer.weight)
            layer.biasZwischenSpeicher = np.zeros_like(layer.bias)
        layer.weightZwischenSpeicher = self.rho * layer.weightZwischenSpeicher + (1. - self.rho) * layer.dweight**2
        layer.biasZwischenSpeicher = self.rho * layer.biasZwischenSpeicher + (1. - self.rho) * layer.dbias**2
        layer.weight -= self._currentLearnRate * layer.dweight / (np.sqrt(layer.weightZwischenSpeicher) + self.epsilon)
        layer.bias -= self._currentLearnRate * layer.dbias / (np.sqrt(layer.biasZwischenSpeicher) + self.epsilon)
    
    def __repr__(self):
        return f"RMSProp(LearningRate={self.lernrate}, learningRateDecay={self._learnRateDecay},rho={self.rho}, epsilon={self.epsilon})"
        
class Adam(Optimizer):
    
    def __init__(self, lernrate=1e-3, lernRateDecay=1e-4, momentum=0.9, beta=0.999, epsilon=1e-7):
        super().__init__(lernrate, lernRateDecay)
        self._momentum, self.beta, self.epsilon = momentum, beta, epsilon
    
    def step(self, layer):
        """
        testing
        """
        if not hasattr(layer, "weightMomentum"):
            layer.weightMomentum = np.zeros_like(layer.weight)
            layer.weightZwischenSpeicher = np.zeros_like(layer.weight)
            layer.biasMomentum = np.zeros_like(layer.bias)
            layer.biasZwischenSpeicher = np.zeros_like(layer.bias)
        layer.weightMomentum = self._momentum * layer.weightMomentum + (1 - self._momentum) * layer.dweight
        layer.biasMomentum = self._momentum * layer.biasMomentum + (1 - self._momentum) * layer.dbias
        weightMomentumKorrigiert = layer.weightMomentum / (1 - self._momentum**(self._stepCount + 1))
        biasMomentumKorrigiert = layer.biasMomentum / (1 - self._momentum**(self._stepCount + 1))
        layer.weightZwischenSpeicher = self.beta * layer.weightZwischenSpeicher + (1 - self.beta) * layer.dweight**2
        layer.biasZwischenSpeicher = self.beta * layer.biasZwischenSpeicher + (1 - self.beta) * layer.dbias**2
        weightZwischenSpeicherKorrigiert = layer.weightZwischenSpeicher / (1 - self.beta**(self._stepCount + 1))
        biasZwischenSpeicherKorrigiert = layer.biasZwischenSpeicher / (1 - self.beta**(self._stepCount + 1))
        layer.weight -= self._currentLearnRate * weightMomentumKorrigiert / (np.sqrt(weightZwischenSpeicherKorrigiert) + self.epsilon)
        layer.bias -= self._currentLearnRate * biasMomentumKorrigiert / (np.sqrt(biasZwischenSpeicherKorrigiert) + self.epsilon)
    
    def __repr__(self):
        return f"Adam(LearningRate={self.lernrate}, learningRateDecay={self._learnRateDecay}, momentum={self._momentum}, beta={self.beta}, epsilon={self.epsilon})"
        