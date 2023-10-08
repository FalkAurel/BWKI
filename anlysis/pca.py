import numpy as np

class PrincipalComponentAnalysis():
    def fit(self, data, nComponents):
        self.mean = data.mean(axis=0, keepdims=True)
        self.meanCenteredData = data - self.mean
        covMatrix = self.covarianceMatrix(self.meanCenteredData)
        self.eigenValues, eigenVectors = np.linalg.eig(covMatrix)
        self.eigenVectors = eigenVectors[:, :nComponents]
    
    def covarianceMatrix(self, data):
        """
        Calculates the covariance matrix. Calculates the variance along the first dimension, assuming that each
        column contains all the corresponding x values. The covariance is the summed product along the last
        dimension in the dataset. The covariance matrix contains for all indices where j == i the variance and
        for all j != i the corresponding covariance.
        """
        return data.T.dot(data) / (len(data) - 1)
    
    @property
    def explainedVariance(self):
        """
        Returns an array which lists all the components and the extend to which they can explain the variance
        found in the dataset. Therefore the sum of all the components should equal 1, hence no information is
        lost.
        """
        return self.eigenValues / self.eigenValues.sum()
        
    def project(self):
        """
        Projects the data into lower-dimensional-space using the principal Components as new axes, it's essentially
        a linear Transformation.
        """
        return self.meanCenteredData.dot(self.eigenVectors)
    
    def inverseTransform(self, data):
        """
        Reverses the linear Transformation and "wraps" the data back into orginal space.
        """
        return data.dot(self.eigenVectors.T) + self.mean
