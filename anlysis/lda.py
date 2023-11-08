import numpy as np

class linearDiscriminantAnalysis():
    def fit(self, X, y, *, nDimensions=2, regularization=1):
        """
        Takes your samples as input, has to be 2D and y which has to be a vector containing all the labels.
        nDimensions specifies how many features the output contains, it is set by default to 2.
        regularization helps to combat the small-sample-size(SSS) Problem, by giving the determinant a bias, thus
        preventing a singular matrix. Please note that it requires fine tuning and when applying the inverse
        Transform no mathematical meaning can be assigned to the regularization term, by default it is set to 1.
        """
        data, self.nDimensions = np.hstack((X, y[:, None])), nDimensions
        self.data, self.mean, (meanMatrix, nSamplesPerClass) = data[:, :-1], data[:, :-1].mean(axis=0), self.findingMeanMatrix(data)
        betweenClassMatrix = self.findingBetweenClassMatrix(meanMatrix,  self.mean, nSamplesPerClass)
        withinClassMatrix = self.findingWithinClassMatrix(data, meanMatrix)
        self.eigenValues, self.eigenVectors = np.linalg.eig(np.linalg.inv(withinClassMatrix * regularization).dot(betweenClassMatrix))
        
    def findingMeanMatrix(self, data):
        """
        Returns a CxF Matrix.
        It contains C vectors of with each of them having a length of F, where F represents the number
        of features present in the dataset.
        """
        uniqueLabels, numberOfSamples = np.unique(data[:, -1], return_counts=True)
        return np.array([data[data[:, -1] == label][:, :-1].mean(axis=0) for label in uniqueLabels]), numberOfSamples
    
    def findingBetweenClassMatrix(self, meanMatrix, overallMean, nSamples):
        """
        Computing the between class scatter matrix. This is done by computing the between class
        variance for every class and summing the weighted results up.
        The class variance for the ith class is given by (u_i.T - u) x (u_i - u).T * n_i.
        u_i represents the ith mean vector.
        n_i represents the number of samples of the ith class.
        """
        nDimensions = meanMatrix.shape[1]
        betweenClassMatrix = np.zeros((nDimensions, nDimensions))
        for meanVector, nSample in zip(meanMatrix, nSamples):
            betweenClassMatrix += nSample * np.outer(meanVector - overallMean, meanVector - overallMean)
        return betweenClassMatrix
    
    def findingWithinClassMatrix(self, data, meanMatrix):
        """
        Computes the within class matrix. This is done by computing the within class variance for
        every class and summing the results up.
        The within class variance is given by sum((x_j_i - u_j).T x (x_j_i - u_j)).
        x_j_i represents the ith element in the jth class.
        u_j represents the mean vector of the jth class.
        """
        uniqueLabels, nDimensions = np.unique(data[:, -1]), data.shape[-1] - 1
        withinClassMatrix = np.zeros((nDimensions, nDimensions))
        for label, meanVector in zip(uniqueLabels, meanMatrix):
            classSpecificData = data[data[:, -1] == label][:, :-1]
            withinClassMatrix += (classSpecificData - meanVector).T.dot(classSpecificData - meanVector)
        return withinClassMatrix
    
    @property
    def project(self):
        """
        Applies the transformation into LDA-Space from orginal space.
        """
        return self.data.dot(self.eigenVectors[:, :self.nDimensions])
    
    def inverseTransform(self, data):
        """
        Performs the inverse Transformation from LDA-Space into original space.
        """
        return data.dot(np.linalg.pinv(self.eigenVectors[:, :self.nDimensions])) + self.mean