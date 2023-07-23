import numpy as np

#Quelle: https://math.stackexchange.com/questions/3323753/how-to-get-the-derivative-of-an-average

class MeanPooling():
    def __init__(self, meanPooling = 2):
        self.meanPoolingSize = meanPooling
    
    def forward(self, image):
        self._input = image
        y, x = image.shape
        image = image[:y - y % self.meanPoolingSize, :x - x % self.meanPoolingSize] 
        return image.reshape(y // self.meanPoolingSize, self.meanPoolingSize, x // self.meanPoolingSize, self.meanPoolingSize).mean(axis = (1, 3))
    
    def backward(self, gradient):
        dInput = np.zeros_like(self._input).astype(np.float64)
        nenner = self.meanPoolingSize**2
        dAverage = np.ones((self.meanPoolingSize, self.meanPoolingSize)) * 1/nenner
        h, w = gradient.shape
        for y in range(h):
            for x in range(w):
                dInput[y:y+self.meanPoolingSize, x:x+self.meanPoolingSize] += dAverage * gradient[y, x]
        return dInput