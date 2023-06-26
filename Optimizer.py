from abc import ABC, abstractmethod

class Optimizer(ABC):
    
    @abstractmethod
    def __init__(self, lernrate):
        self.lernrate = lernrate
    
    @abstractmethod
    def step(self, layer):
        pass

class SGD(Optimizer):
    def __init__(self, lernrate):
        super().__init__(lernrate)
    
    def step(self, layer):
        layer.weight -= layer.dweight * self.lernrate
        layer.bias -= layer.dbias * self.lernrate