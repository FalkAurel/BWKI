import numpy as np

class DenseLayer(object):
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
        inputs = inputs.reshape(self._inputShape[0], -1)
        return inputs
    
    def backward(self, inputs):
        inputs = np.squeeze(inputs)
        return inputs.reshape(self._inputShape)
    
class BatchNormalizationLayer():
    """
    Improves training performance by reparametrising the underlying optimization problem. BatchNormalization
    smoothens out the gradient, enabling gradient-based learning to take larger steps. -> Increases stability and makes
    it less dependant on the choice of hyperparameters.
    """
    def __init__(self):
        """
        Please note that I'm substituting gamma and beta for weight and bias to make this module compa-
        tible with the rest of the libary.
        """
        self.weight = 1
        self.bias = 0
        self.l1WFaktor = self.l1BFaktor = self.l2WFaktor = self.l2BFaktor = 0#this is implemented so that the model can be built.
        
    def forward(self, inputs):
        """
        Subtracting from the input its mean before dividing by the standard deviation of the input.
        Finally multiplying it by the self.weight parameter and adding self.bias to it.
        """
        self.inputs = inputs
        self.mean = np.mean(inputs, axis=0)
        self.variance = np.var(inputs, axis=0)
        self.stdDev = np.sqrt(self.variance + 1e-8)
        self.normalizedInputs = (inputs - self.mean) / self.stdDev
        return self.weight * self.normalizedInputs + self.bias
    
    def backward(self, gradient):
        """
        Backpropagation through the layer. We first compute the gradients of the loss with respect to
        the normalized inputs, variance, and mean. Then we apply the chain rule to derive dweight,
        dInput and dbias. As per usual, dbias is just the gradient as its derivative of the sum op-
        eration is one.
        """
        N, D = gradient.shape
        dNormalizedInputs = gradient * self.weight
        dVariance = np.sum(dNormalizedInputs * (self.inputs - self.mean) * -0.5 * (self.variance + 1e-8)**(-1.5), axis=0)
        dMean = np.sum(dNormalizedInputs * -1 / self.stdDev, axis=0) + dVariance * np.mean(-2 * (self.inputs - self.mean), axis=0)
        dInput = dNormalizedInputs / self.stdDev + dVariance * 2 * (self.inputs - self.mean) / N + dMean / N
        self.dweight = np.sum(gradient * self.normalizedInputs, axis=0)
        self.dbias = np.sum(gradient, axis=0)
        return dInput
