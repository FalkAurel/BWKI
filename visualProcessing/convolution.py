import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import scipy.signal

#Quelle: https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c

class Convolution():
    def __init__(self, inputGröße:tuple ,*,kernelGröße = 2):
        """
        Initilize Class
        """
        self.kernelGröße = kernelGröße
        self.weight = np.random.randn(inputGröße[0], kernelGröße, kernelGröße) / kernelGröße**2
        self.outputShape = (inputGröße[0], inputGröße[1] - kernelGröße + 1, inputGröße[2] - kernelGröße + 1)
        self.bias = np.random.randn(*self.outputShape)
    
    def forward(self, image):
        """
        Es wird eine Kopie vom Bild erzeugt, damit man es später in dem BackwardPass benutzen kann, um die dweights zu err
        echnen und es als Vorlage für den dInput zu nutzen.
        Forwardpass valide Kreuzkorrelation am Bild. Es wird eine featureMap erstellt. 
        """
        self.input = image
        subArray = sliding_window_view(image, window_shape=(self.weight.shape))
        featureMap = np.sum(self.weight * subArray, axis =(3, 4, 5))#entfernt 3 dimensionen -> 6dim wird zu 3dim
        return featureMap + self.bias
    
    def forwardOptim(self, image):
        """
        Es wird eine Kopie vom Bild erzeugt, damit man es später in dem BackwardPass benutzen kann, um die dweights zu err
        echnen und es als Vorlage für den dInput zu nutzen.
        Forwardpass valide Kreuzkorrelation am Bild. Es wird eine featureMap erstellt. 
        """
        self.input = image
        featureMap = np.zeros(self.outputShape)
        for i in range(image.shape[0]):
            subArray = sliding_window_view(image[i], window_shape=self.weight.shape[1:])
            featureMap[i] = np.sum(self.weight[i] * subArray, axis=(2, 3))
        return featureMap + self.bias

    
    def backward(self, gradient):
        """
        BackwardPass wird durchgeführt.
        dInput ist nichts anderes als die 90° rotiertete volle Kreuzkorrelation zwischen der weight Matrix und dem
        Gradienten
        Die Ableitung einer Summe ist 1, weshalb der Gradient direkt für dbias übernommen wird
        nach Kettenregel(1 * gradient).
        """
        dInput = np.copy(self.input).astype(np.float64)
        self.dbias = gradient
        self.dweight = np.zeros(self.weight.shape)
        for batch in range(len(gradient)):
            subArrays = sliding_window_view(self.input[batch], window_shape=(gradient[batch].shape))
            self.dweight[batch] = np.sum(subArrays * gradient, axis = (2, 3))
            dInput[batch] = scipy.signal.convolve2d(np.flip(self.weight[batch]), gradient[batch], mode="full")
        return dInput

if __name__ == "__main__":
    conv = Convolution((1, 300, 300))
    test = np.random.randn(1, 300, 300)
    """test = np.array([[[1, 6, 2],
                      [5, 3, 1],
                      [7, 0, 4]]])
    
    conv.weight = np.array([[[1, 2],
                             [-1, 0]]])"""
    from time import perf_counter
    start = perf_counter()
    out = conv.forwardOptim(test)
    gradient = np.random.randn(*out.shape)
    dInput = conv.backward(gradient)
    ende = perf_counter()
    print(ende - start)
