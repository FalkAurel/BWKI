from abc import ABC, abstractmethod, abstractclassmethod
import numpy as np

class Optimizer(ABC):
    
    @abstractmethod
    def __init__(self, lernrate, momentum):
        self._lernrate = lernrate
        self._stepCount = 1
        self._momentum = momentum
        
    @abstractmethod
    def step(self, layer):
        pass
    
    @abstractmethod
    def learningRateDecay(self):
        pass
    
    @abstractclassmethod
    def setLearnRateDecay(cls, value):
        pass
    
    @abstractmethod
    def __repr__(self):
        pass
    
class SGD(Optimizer):
    
    _learnRateDecay = 1e-7
    
    def __init__(self, lernrate, momentum):
        super().__init__(lernrate, momentum)
        
    def step(self, layer):
        if self._momentum:
            if not hasattr(layer, "_weightMomentum"):
                layer._weightMomentum = np.zeros_like(layer._weight)
                layer._biasMomentum = np.zeros_like(layer._bias)
            updateWeight = self._momentum * layer._weightMomentum - self._lernrate * layer._dweight
            updateBias = self._momentum * layer._biasMomentum - self._lernrate * layer._dbias
            layer._weightMomentum = updateWeight
            layer._biasMomentum = updateBias
            layer._weight += updateWeight
            layer._bias += updateBias
        else:
            layer._weight -= self._lernrate * layer._dweight
            layer._bias -= self._lernrate * layer._dbias
        return self
    
    def learningRateDecay(self):
        self._lernrate = self._lernrate * 1. / (1. + self._learnRateDecay * self._stepCount)
        self._stepCount += 1
    
    def __repr__(self):
        return f"SGD(LearningRate={self._lernrate}, learningRateDecay={self._learnRateDecay}, momentum={self._momentum})"
    
    @classmethod
    def setLearnRateDecay(cls, value):
        cls._learnRateDecay = value
