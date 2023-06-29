import numpy as np

class Metrics(object):
    
    @staticmethod
    def accuracy(yhat, y):
        if y.ndim == 2:
            y = y.argmax(axis = 1)
        return np.mean(yhat.argmax(axis = 1) == y)
