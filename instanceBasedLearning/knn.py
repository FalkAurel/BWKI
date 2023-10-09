import numpy as np

class KNN():
    def fit(self, X, y, k):
        self.data = np.hstack((X, y.reshape(-1, 1)))
        self.k = k
        return self
    
    def predict(self, X, mode = "classification"):
        """
        Calculates the inverse weighted output. The closer, the bigger essentially, hence the significance for close
        points increases. 
        """
        distanceVector = self.euclideanDistanceToTargetPoint(X)
        neighbours, weights = self.findNearestNeighbours(distanceVector)
        if mode == "regression":
            return np.average(self.data[neighbours][:, -1], weights=1 / (np.array(weights) + 1e-9)).round()
        return self.majorityVoting(self.data[neighbours][:, -1], weights)
    
    def evaluate(self, X, y, *, mode = "classification", predictions = False, tolerance = 0.05):
        """
        Evaluates a batch of inputs. By default the mode is set to "classification" for regression tasks just set it to
        "regression". Furthermore it returns the percentage of correct predictions; the accuracy. If you'd like to get
        the predictions made set predictions to "True". The tolerance parameter is only used for the regression Mode. It's
        set to a 5% tolerance by default.
        """
        yhat = [self.predict(x, mode=mode) for x in X]
        if predictions:
            return yhat
        if mode == "regression":
            return np.mean(np.abs(y - yhat) <= np.abs(y - y * (1 + tolerance)))
        return np.mean(yhat == y)
        
    def euclideanDistanceToTargetPoint(self, target):
        """
        Calculates the distance between the target point and all the points present in the train dataset using euclidean
        distance. returns a vector containing all the distances.
        """
        cords = self.data[:, :-1]
        return np.square(cords - target).sum(axis = -1)**0.5
    
    def findNearestNeighbours(self, distance):
        """
        Using a greedy search algorithm to find the k nearest neighbours.
        """
        neighbours, distanceToTarget = [], []
        distance = distance.copy()
        for i in range(self.k):
            position = distance.argmin()
            distanceToTarget.append(distance[position])
            neighbours.append(position)
            distance[position] = np.inf
        return neighbours, distanceToTarget
    
    def majorityVoting(self, y, weight):
        inverseWeight = 1 / (np.array(weight) + 1e-9)
        uniqueLabel = np.unique(y)
        weightedY = np.array([np.sum(np.where(y == label, 1, 0) * inverseWeight) for label in uniqueLabel])
        return uniqueLabel[np.argmax(weightedY)]
