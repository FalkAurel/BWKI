import numpy as np
from abc import ABC, abstractmethod

class LossFunction(ABC):
    
    @abstractmethod
    def calculate(self, yhat, y):
        pass
    
    @abstractmethod
    def loss(self):
        pass
    
class CategoricalCrossEntropyLoss(LossFunction):
    
    def calculate(self, yhat, y):
        """
        Edit: np.clip(yhat[y.argmax()], 1e-7, 1-1e-7) handelt unsere Edge-Cases wo wir einen loge(0) machen würden.
        Da e**x immer y > 0 ist kann der 0 log nicht existieren also clippen wir das ganze. Andernfalls würden wir bei
        einer 1 -> das Model ist sich 100% sicher anfangen das ganze ins negative shiften, weshalb np.log(wert + kleine
        Zahl) nicht funktionieren würde.
        """
        if yhat.shape[0] == 1 and y.shape[0] == 1:
            yhat, y = yhat.reshape(-1), y.reshape(-1)
        self._loss = np.log(np.clip(yhat[y.argmax()], 1e-7, 1-1e-7)) * -1
        return self
    
    def loss(self):
        return np.mean(self._loss)
    
    def backward(self, yhat, y):
        sample = len(yhat)
        if len(y.shape) == 1:
            y = np.eye(len(yhat[0]))[y]
        self.output = -y / yhat
        self.output = self.output / sample
        return self.output
    
if __name__ == "__main__":
    softmaxOutput = np.array([[0.7, 0.1, 0.2],
                          [0.8, 0.15, 0.05],
                          [0.5, 0.27, 0.23]])

    groundTruth = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
    loss = CategoricalCrossEntropyLoss()
    loss.calculate(softmaxOutput, groundTruth)
    print(loss.loss())