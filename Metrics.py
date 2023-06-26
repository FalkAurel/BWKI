import numpy as np

class Metrics(object):
    
    @staticmethod
    def accuracy(yhat, y):
        return f"{np.mean(yhat.argmax(axis = 1) == y.argmax(axis = 1))}%"
    
if __name__ == "__main__":
    a = np.array([[0.7, 0.2, 0.1],
                  [0.5, 0.1, 0.4],
                  [0.02, 0.9, 0.08]])
    b = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 1, 0]])
