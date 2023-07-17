import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

#Quelle https://victorzhou.com/blog/intro-to-cnns-part-1/
#Xavier Initialization: https://www.quora.com/What-is-an-intuitive-explanation-of-the-Xavier-Initialization-for-Deep-Neural-Networks

class Convolution(object):
    def __init__(self, inputGröße, *, kernelGröße = 2, filterNum = 16):
        self.kernelGröße = kernelGröße
        self.weight = np.random.randn(filterNum, kernelGröße, kernelGröße) / kernelGröße**2# Xavier Initialization
        self.outputShape = (inputGröße[0] - kernelGröße + 1, inputGröße[1] - kernelGröße + 1)
        self.bias = np.random.randn(*self.outputShape)
    
    def forward(self, image):
        self.input = image
        subArrays = sliding_window_view(image, window_shape=self.weight.shape)
        featureMap = np.tensordot(subArrays, self.weight, axes=([2,3],[0,1]))
        return featureMap + self.bias
    
    def backward(self, gradient):
        raise NotImplementedError
    
if __name__ == "__main__":
    conv = Convolution((3, 3))
    conv.weight = np.array([[1, 2,],
                            [-1, 0]])
    testMatrix = np.array([[1, 6, 2],
                           [5, 3, 1],
                           [7, 0, 4]])
    out = conv.forward(testMatrix)
