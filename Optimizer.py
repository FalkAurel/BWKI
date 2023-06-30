import numpy as np

class Optimizer(object):
    _learnRateDecay = 1e-3
    
    def __init__(self, lernrate):
        self.lernrate = lernrate
        self._stepCount = 0
        self._currentLearnRate = lernrate
        
    def step(self, layer):
        raise NotImplementedError
    
    def learningRateDecay(self):
        self._currentLearnRate = self.lernrate / (1. + self._stepCount * self._learnRateDecay)
        self._stepCount += 1
    
    @classmethod
    def setLearnRateDecay(cls, value):
        cls._learnRateDecay = value
    
    def __repr__(self):
        raise NotImplementedError

    
class SGD(Optimizer):
    """
    Stochastic Gradient Descent(SGD). Vanilla-Optimizer. Dieser hier supported learningRateDecay; kontrollierte Ab-
    nahme der Learning Rate und Momentum. Momentum ist echt interessant, da wir so aus einem lokalen Minimum(leicht-
    er / eher) ausbrechen können. Momentum ist nur das vorherrige Update zu dem Parameter mit einem Faktor zwischen
    0 - 1; wieviel davon übernommen werden soll.
    """
    _learnRateDecay = 1e-3
    
    def __init__(self, lernrate, momentum):
        super().__init__(lernrate)
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
    _learnRateDecay = 1e-5
    
    def __init__(self, lernrate, epsilon = 1e-7):
        super().__init__(lernrate)
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
