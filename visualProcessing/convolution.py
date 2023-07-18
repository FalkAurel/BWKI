import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import scipy.signal

#Quelle https://victorzhou.com/blog/intro-to-cnns-part-1/
#Xavier Initialization: https://www.quora.com/What-is-an-intuitive-explanation-of-the-Xavier-Initialization-for-Deep-Neural-Networks
#https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c

class Convolution(object):
    def __init__(self, inputGröße, *, kernelGröße = 2):
        self.kernelGröße = kernelGröße
        self.weight = np.random.randn(kernelGröße, kernelGröße) / kernelGröße**2# Xavier Initialization
        self.outputShape = (inputGröße[0] - kernelGröße + 1, inputGröße[1] - kernelGröße + 1)
        self.bias = np.random.randn(*self.outputShape)
    
    def forward(self, image):
        self.input = image
        subArrays = sliding_window_view(image, window_shape=self.weight.shape)
        featureMap = np.tensordot(subArrays, self.weight, axes=([2,3],[0,1]))
        return featureMap + self.bias
    
    def backward(self, gradient):
        subArrays = sliding_window_view(self.input, window_shape=gradient.shape)
        self.dweight = np.tensordot(subArrays, gradient, axes=([2, 3], [0, 1]))
        self.dbias = gradient
        dInput = scipy.signal.convolve2d(np.flip(self.weight), gradient, mode="full")
        return dInput
