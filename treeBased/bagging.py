import numpy as np

"""
This modul is basically just a wrapper for the decisionTree modul, it just adds a few extra functionalities.
"""
class Bagging():
    """
    Takes a list of decisionTrees which are used to make predictions
    """
    def __init__(self, Trees:list, mode="classification"):
        self.trees = Trees
        self.mode = mode
    
    def bootstrapAggregating (self, X, y, batchSize):
        """
        creates a randomly sampled dataset
        """
        outputX, outputY = [], []
        numSamples = X.shape[0]
        for _ in range(len(self.trees)):
            index = np.random.randint(0, numSamples, size=batchSize)
            newY = y[index]
            newX = X[index][:, np.random.choice(X.shape[1], size=X.shape[1], replace=True)]
            outputX.append(newX)
            outputY.append(newY)
        return np.array(outputX), np.array(outputY)

    def fit(self, X, Y):
        for tree, x, y in zip(self.trees, X, Y):
            tree.fit(x, y, mode=self.mode)
    
    def evaluate(self, X):
        yhat = []
        for tree in self.trees:
            yhat.append(tree.predict(X, tree.root))
        yhat = np.array(yhat)
        if self.mode == "regression":
            return np.mean(yhat)
        return np.argmax(np.bincount(yhat.astype(np.int64)))