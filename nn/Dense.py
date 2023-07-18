import numpy as np

class Layer(object):
    pass

class DenseLayer(Layer):
    def __init__(self, n_input, n_neuronen, l1WFaktor=0, l1BFaktor=0, l2WFaktor=0, l2BFaktor=0):
        self.weight = np.random.randn(n_input, n_neuronen) * 1e-3
        self.bias = np.zeros((1, n_neuronen))
        self.l1WFaktor, self.l1BFaktor = l1WFaktor, l1BFaktor
        self.l2WFaktor, self.l2BFaktor = l2WFaktor, l2BFaktor
        
    def forward(self, inputs):
        self.input = inputs
        return np.dot(inputs, self.weight) + self.bias
    
    def backward(self, gradient):
        """
        forwardPass = sum(mul(x[0, 0], w[0, 0]), ...mul(x[i,j], w[i,j]), b)
        dsum/dx = f(x, y) = x + y -> f'(x) = 1*x^1-1 = 1 * 1 = 1
        dmul/dx = f(x, y) = x * y -> f'(x) = y * 1*x^1-1 = y * 1 = y
        """
        dInput = np.dot(gradient, self.weight.T)
        self.dweight = np.dot(self.input.T, gradient)
        self.dbias = gradient.sum(axis = 0, keepdims = True)
        if self.l1WFaktor:
            self.dweight += self.l1WFaktor * np.where(self.weight >= 0, 1, -1)
        if self.l1BFaktor:
            self.dbias += self.l1BFaktor * np.where(self.bias >= 0, 1, -1)
        if self.l2WFaktor:
            self.dweight += 2 * self.l2WFaktor * self.weight
        if self.l2BFaktor:
            self.dbias += 2 * self.l2BFaktor * self.bias
        return dInput
    
    @property
    def getParams(self):
        return self.weight, self.bias, self.l1WFaktor, self.l1BFaktor, self.l2WFaktor, self.l2BFaktor
    
    def setParams(self, weight, bias, l1WFaktor, l1BFaktor, l2WFaktor, l2BFaktor):
        self.weight, self.bias, self.l1WFaktor, self.l1BFaktor, self.l2WFaktor, l2BFaktor = weight, bias, l1WFaktor, l1BFaktor, l2WFaktor, l2BFaktor
        
    def __repr__(self):
        return f"DenseLayer(weightTensorShape={self.weight.shape}, biasTensorShape={self.bias.shape})"

class DropOutLayer(object):
    def __init__(self, dropOutChance):
        self.dropOutChance = dropOutChance
    
    def forward(self, inputs):
        self._binaryMask = np.random.binomial(1, 1 - self.dropOutChance, size=inputs.shape)
        return inputs * self._binaryMask
    
    def backward(self, gradient):
        return gradient * self._binaryMask

class FlattenLayer(object):
    def forward(self, inputs):
        self._inputShape = inputs.shape
        return inputs.reshape(-1)
    
    def backward(self, inputs):
        return inputs.reshape(self._inputShape)
