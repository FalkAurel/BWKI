import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import scipy.signal

class Convolution():
    def __init__(self, inputGröße:tuple ,*, kernelGröße = 2, batchSize = 64):
        self.kernelGröße = kernelGröße
        self.weight = np.random.randn(inputGröße[0], kernelGröße, kernelGröße) / kernelGröße**2
        self.outputShape = (inputGröße[0], inputGröße[1] - kernelGröße + 1, inputGröße[2] - kernelGröße + 1)
        self.bias = np.random.randn(*self.outputShape)
        self.batchSize = batchSize
        self.l1WFaktor = self.l1BFaktor = self.l2WFaktor = self.l2BFaktor = 0
    
    def forward(self, image):
        """
        We create a copy of the image. The forwardPass is nothing more than the cross-correlation between the weights and
        the image, resulting in a small part of the featureMap.
        """
        self.input = image
        featureMap = np.zeros(self.outputShape)
        for i in range(image.shape[0]):
            subArray = sliding_window_view(image[i], window_shape=self.weight.shape[1:])
            featureMap[i] = np.sum(self.weight[i] * subArray, axis=(2, 3))
        return featureMap + self.bias
    
    def backward(self, gradient):
        """
        dInput is the 90° rotated full cross-correlation between the weight and the gradient, resulting in a matrix of
        shape image. dweight is the cross-correlation between the input and the gradient. dbias is just the gradient.
        """
        dInput = np.copy(self.input).astype(np.float64)
        self.dbias = gradient.sum(axis = 0, keepdims = True) / self.batchSize
        self.dweight = np.zeros(self.weight.shape)
        for batch in range(len(gradient)):
            subArrays = sliding_window_view(self.input[batch], window_shape=(gradient[batch].shape))
            self.dweight[batch] = np.sum(subArrays * gradient[batch], axis = (2, 3))
            dInput[batch] = scipy.signal.convolve2d(gradient[batch], self.weight[batch], mode="full")
        self.dweight /= self.batchSize
        return dInput