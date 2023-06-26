import numpy as np

class Metrics(object):
    
    @staticmethod
    def accuracy(yhat, y):
        return f"{np.mean(yhat.argmax(axis = 1) == y.argmax(axis = 1))}%"
