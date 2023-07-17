import numpy as np

#Quellen https://numpy.org/doc/stable/reference/generated/numpy.mgrid.html

class MaxPooling(object):
    def __init__(self, maxPoolingSize):
        self.maxPoolingSize = maxPoolingSize
    
    def forward(self, image):
        self._input = image
        h, w  = image.shape
        output = np.zeros((h // self.maxPoolingSize, w // self.maxPoolingSize))
        self._maxOutputIndeces = np.copy(output)
        for y in range(0, h - h % self.maxPoolingSize, self.maxPoolingSize):
            for x in range(0, w - w % self.maxPoolingSize, self.maxPoolingSize):
                Y, X = y // self.maxPoolingSize, x // self.maxPoolingSize
                output[Y, X] = np.max(image[y:y+self.maxPoolingSize, x:x+self.maxPoolingSize])
                self._maxOutputIndizes[Y, X] = np.argmax(image[y:y+self.maxPoolingSize, x:x+self.maxPoolingSize])
        return output
    
    """
    Funktioniert nicht, Optimierung wurde noch nicht implementiert.
    def forwardOptim(self, image):
        self._input = image
        h, w = image.shape
        self._maxOutputIndeces1 = np.copy(np.zeros((h // self.maxPoolingSize, w // self.maxPoolingSize)))
        y, x = image.shape
        y //= self.maxPoolingSize
        x //= self.maxPoolingSize
        self._maxOutputIndizes = image[:y*self.maxPoolingSize, :x*self.maxPoolingSize].reshape(y, self.maxPoolingSize, x, self.maxPoolingSize).argmax(axis = 1).argmax(axis = 2)
        return image[:y*self.maxPoolingSize, :x*self.maxPoolingSize].reshape(y, self.maxPoolingSize, x, self.maxPoolingSize).max(axis = (1, 3))
    """
    def backward(self, gradient):
        dInput = np.zeros_like(self._input)
        h, w = self._input.shape
        Y, X = np.mgrid[0:h - h % self.maxPoolingSize:self.maxPoolingSize, 0:w - w % self.maxPoolingSize:self.maxPoolingSize]
        Y //= self.maxPoolingSize
        X //= self.maxPoolingSize
        maxIndex = np.unravel_index(self._maxOutputIndeces[Y, X].astype(int), (self.maxPoolingSize, self.maxPoolingSize))
        dInput[Y * self.maxPoolingSize + maxIndex[0], X * self.maxPoolingSize + maxIndex[1]] = gradient[Y, X]
        return dInput

if __name__ == "__main__":
    from time import perf_counter
    m = MaxPooling(2)
    test = np.array([[1, 2, 3 ,4],
                     [5, 1, 2, 3],
                     [4, 5, 1, 2],
                     [3, 4, 5, 1]])
    gradient = np.array([[2, -3],
                         [4, 5]])
    out = m.forward(test)
    dInput = m.backward(gradient)
    """
    expected Output
    [[ 0,  0,  0, -3],
    [ 2,  0,  0,  0],
    [ 0,  4,  0,  0],
    [ 0,  0,  5,  0]]
    """
    