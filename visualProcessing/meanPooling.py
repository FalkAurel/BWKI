import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

#Quelle: https://math.stackexchange.com/questions/3323753/how-to-get-the-derivative-of-an-average

class MeanPooling():
    def __init__(self, meanPooling = 2):
        self.meanPoolingSize = meanPooling
    
    def forward(self, image):
        self._input = image
        _, y, x = image.shape
        image = image[:y - y % self.meanPoolingSize, :x - x % self.meanPoolingSize] 
        return image.reshape(-1, y // self.meanPoolingSize, self.meanPoolingSize, x // self.meanPoolingSize, self.meanPoolingSize).mean(axis = (2, 4))
    
    def backward(self, gradient):
        dInput = np.zeros_like(self._input).astype(np.float64)
        nenner = self.meanPoolingSize**2
        dAverage = np.ones((self.meanPoolingSize, self.meanPoolingSize)) * 1/nenner
        batch, h, w = gradient.shape
        for b in range(batch):
            for y in range(h):
                for x in range(w):
                    dInput[b, y:y+self.meanPoolingSize, x:x+self.meanPoolingSize] += dAverage * gradient[b, y, x]
        return dInput