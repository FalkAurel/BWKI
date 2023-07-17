from tqdm import tqdm
from .ActivationFunction import ActivationFunction
import pickle
import copy

class Model(object):
    def __init__(self):
        self.layer = []
        self._tranierbareLayer, self._final = [], []
    
    def add(self, layer):
        if hasattr(layer, "weight"):
            self._tranierbareLayer.append(layer)
        if issubclass(layer.__class__, ActivationFunction) or hasattr(layer, "weight"):
            self._final.append(layer)
        self.layer.append(layer)
        
    def Set(self, *, loss, optimizer, metrics):
        self.loss, self.optim, self.metrics = loss, optimizer, metrics
        
    def _forward(self, inputs):
        for layer in self.layer:
            inputs = layer.forward(inputs)
        return inputs
    
    def _backward(self, output):
        for layer in reversed(self.layer):
            output = layer.backward(output)
    
    def _optim(self):
        self.optim.learningRateDecay()
        for layer in self._tranierbareLayer:
            self.optim.step(layer)
    
    def predict(self, X):
        for layer in self._final:
            X = layer.forward(X)
        return X
        
    def train(self, X, y, *, epochs = 1, interval = 0.1, tracking=False, batch=1):
        acc, loss, lr = [], [], []
        check = int(epochs * interval)
        steps = len(X) // batch
        if tracking:
            self.loss.tranierbareLayer(self._tranierbareLayer)
        for epoch in tqdm(range(epochs)):
            accCounter = 0
            lossCounter = 0
            counter = 0
            for step in range(steps):
                batchX = X[step * batch:(step + 1)*batch]
                batchY = y[step * batch:(step + 1)*batch]
                output = self._forward(batchX)
                self._backward(self.loss.backward(output, batchY))
                self._optim()
                if tracking and epoch % check == 0:
                    acc.append(self.metrics(output, batchY))
                    loss.append(self.loss.calculate(output, batchY) + self.loss.regularizationLoss())
                    lr.append(self.optim.getLearningRate)
        return acc, loss, lr, self.optim

    def _getParams(self):
        parameter = []
        for layer in self._tranierbareLayer:
            parameter.append(layer.getParams)
        return parameter
    
    def _loadParams(self, parameters, layers):
        for parameter, layer in zip(parameters, layers):
            layer.setParams(*parameter)
    
    def loadParams(self, pfad):
        with open(pfad, "rb") as f:
            self._loadParams(pickle.load(f))
            
    def getParams(self, pfad):
        with open(pfad, "wb") as f:
            pickle.dump(self._getParams(), f)
        
    def freeze(self, pfad):
        model = copy.deepcopy(self)
        for layer in model.layer:
            for property in ["dweight, dbias"]:
                layer.__dict__.pop(property, None)
        with open(pfad, "wb") as f:
            pickle.dump(model, f)
    
    @staticmethod
    def load(pfad):
        with open(pfad, "rb") as f:
            return pickle.load(f)