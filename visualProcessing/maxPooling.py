import numpy as np

#https://www.w3resource.com/numpy/array-creation/mgrid.php
#https://lanstonchu.wordpress.com/2018/09/01/convolutional-neural-network-cnn-backward-propagation-of-the-pooling-layers/

class MaxPooling():
    def __init__(self, *, maxPoolingSize=2):
        self.maxPoolingSize = maxPoolingSize
    
    def forward(self, image):
        """
        We create a Copy of the Image's shape to use it afterwards in the backward Pass. We reshape the image with
        respect to the self.maxPoolingSize so that we get subArrays which we'll sum up along axes -3 and -1 to
        get all the max. Values of the reshaped Array.
        The last step is to create a binary mask that contains all the indices of the max Values of the reshaped Array.
        """
        self._input = image.shape
        b,h, w  = image.shape
        angepasstesFenster = image[:, :int(h - h % self.maxPoolingSize), :int(w - w % self.maxPoolingSize)]
        output = angepasstesFenster.reshape(b, h // self.maxPoolingSize, self.maxPoolingSize, w // self.maxPoolingSize, self.maxPoolingSize).max(axis = (-3, -1))
        self.binaryMask = np.zeros((b, h // self.maxPoolingSize, w // self.maxPoolingSize))
        for batch in range(b):
            for y in range(0, int(h - h % self.maxPoolingSize), self.maxPoolingSize):
                for x in range(0, int(w - w % self.maxPoolingSize), self.maxPoolingSize):
                    self.binaryMask[batch, y // self.maxPoolingSize, x // self.maxPoolingSize] = np.argmax(image[batch, y:y+self.maxPoolingSize, x:x+self.maxPoolingSize])
        return output
    
    def backward(self, gradient):
        """
        We create an Array of shape self._input. Using self.binaryMask we figure out the position of all values which
        have contributed to the output. We set those positions to 1 whereas the remaining values are set to 0. This is
        because if we take the partial derrivative of the max() operation[f(x) = max(x)] we get one value. So the partial
        derrivative would be f'(x) = 1 if x == max(x), otherwise 0.
        Now we can apply the chain rule.
        """
        dInput = np.zeros(self._input)
        b, h, w = self.binaryMask.shape
        b, h, w = dInput.shape
        yIndices, xIndices = np.indices((h // self.maxPoolingSize, w // self.maxPoolingSize))
        yIndices = yIndices * self.maxPoolingSize + self.binaryMask // self.maxPoolingSize
        xIndices = xIndices * self.maxPoolingSize + self.binaryMask % self.maxPoolingSize
        dInput[np.arange(b)[:, None, None], yIndices.astype(int), xIndices.astype(int)] = gradient
        return dInput
