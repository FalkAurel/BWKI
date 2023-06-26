import numpy as np

class DenseLayer(object):
    def __init__(self, n_input, n_neuronen):
        self.weight = np.random.randn(n_input, n_neuronen) * 1e-3
        self.bias = np.zeros((1, n_neuronen))
    
    def forward(self, inputs):
        self.input = inputs
        return np.dot(inputs, self.weight) + self.bias
    
    def backward(self, gradient):
        """
        Backpropagation implementiert mit der Chain Rule. Das Updaten der Parameter findet hier statt. Die Lernrate ist
        variabel.
        Hier gilt: dActivationFunction / dsum * dsum / dmul * dmul / dVARIABLE(z.b.: x0 oder x2 etc.)
        """
        dInput = np.dot(gradient, self.weight.T)
        self.dweight = np.dot(self.input.T, gradient)
        self.dbias = np.sum(gradient, axis = 0, keepdims = True)
        return dInput
    
    def __repr__(self):
        return f"DenseLayer(weightTensorShape={self.weight.shape}, biasTensorShape={self.bias.shape})"
    