import numpy as np

def fourierFeatureMapping(data, shape):
    """
    Takes a matrix as input and a shape tuple, which defines the size of the random gaussian matrix.
    The last dimension of the filter has to match up with first dimension of the data.
    """
    randomGaussianMatrix = 2 * np.pi * np.random.randn(*shape) @ data
    return np.vstack([np.cos(randomGaussianMatrix), np.sin(randomGaussianMatrix)])