import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

#Quelle: https://math.stackexchange.com/questions/3323753/how-to-get-the-derivative-of-an-average

class MeanPooling():
    def __init__(self, meanPooling = 2):
        self.meanPoolingSize = meanPooling
    
    def forward(self, image):
        """
        We store the image's shape in self._input to use it in the backwardPass. We split the image into subArrays
        and take the mean along axes 2 and 4.
        """
        self._input = image.shape
        b, y, x = image.shape
        image = image[:, :y - y % self.meanPoolingSize, :x - x % self.meanPoolingSize] 
        return image.reshape(b, y // self.meanPoolingSize, self.meanPoolingSize, x // self.meanPoolingSize, self.meanPoolingSize).mean(axis = (2, 4))
    
    def backward(self, gradient):
        """
        We create an Array of shape self._input. Now we take the partial derivative of the meanPoolingOperation.
        If u = 1 / m * ∑_(i=1)^m = xi then is du/dx = 1/m. So we can say we take for every value in the gradient only the
        mth part which would look like this gradient/m. We have to iterate over the gradient and apply this gradient to a
        matrix that is of size m**2 and multiply this matrix by gradientElement over m before integrating it into the
        dInput array.
        """
        dInput = np.zeros(self._input)
        nenner = self.meanPoolingSize**2
        dAverage = np.ones((self.meanPoolingSize, self.meanPoolingSize)) * 1/nenner
        batch, h, w = gradient.shape
        for b in range(batch):
            for y in range(h):
                for x in range(w):
                    dInput[b, y:y+self.meanPoolingSize, x:x+self.meanPoolingSize] += dAverage * gradient[b, y, x]
        return dInput