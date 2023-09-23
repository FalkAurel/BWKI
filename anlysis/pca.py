import numpy as np

class PrincipalComponentAnalysis():
    
    def fit(self, data, *, nComponents=1):
        """
        d should never be larger than the dimensions present in the dataset.
        Takes the dataset and shifts it so that it is mean centered.
        """
        self.mean = data.mean(axis=0, keepdims=True)
        self.meanCenteredData = data - self.mean
        eigenValue, self.eigenVector = np.linalg.eig(self.covarianceMatrix())
        self.index = self.findBestEigenVector(eigenValue, nComponents)
        return self
        
    def covarianceMatrix(self):
        """
        Calculates the covariance matrix. Calculates the variance along the first dimension, assuming that each
        column contains all the corresponding x values. The covariance is the summed product along the last
        dimension in the dataset. The covariance matrix contains for all indices where j == i the variance and
        for all j != i the corresponding covariance.
        """
        return np.dot(self.meanCenteredData.T, self.meanCenteredData) / (len(self.meanCenteredData) - 1)
    
    def findBestEigenVector(self, l, dimensions):
        output, currentBest = [], -float("inf")
        for _ in range(dimensions):
            index = np.argmax(l)
            l[index] = -np.inf
            output.append(index)
        return output
    
    def project(self):
        return np.dot(self.meanCenteredData, self.eigenVector[:, self.index])
    
    def inverseTransform(self, data):
        """
        To obtain the untwisted version of the data just take the eigenVector matrix and take the transpose of it
        and take the dot product of it with the data you wish to see in original space.
        """
        return np.dot(data, self.eigenVector[:, self.index].T) + self.mean