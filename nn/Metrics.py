import numpy as np

class Metrics(object):
    
    @staticmethod
    def accuracyClassifier(yhat, y):
        if y.ndim == 2:
            y = y.argmax(axis = 1)
        return np.mean(yhat.argmax(axis = 1) == y)
    
    @staticmethod
    def accuracyRegression(yhat, y, fehlerToleranz = 250):
        fehlerToleranz = np.std(y) / fehlerToleranz
        return np.mean(np.abs(yhat - y) < fehlerToleranz)
 
 