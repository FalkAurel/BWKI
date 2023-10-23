import numpy as np

class KMeansClustering():
    def fit(self, data, *, k=2):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.data = data
        centroids = data[np.random.choice(range(len(data)), k)]
        previousCentroids = np.zeros_like(centroids)
        while not (previousCentroids == centroids).all():
            previousCentroids = centroids.copy()
            clusters = self.assignToClusters(centroids)
            if not all(clusters):
                centroids = data[np.random.choice(range(len(data)), k)]
                continue
            centroids = self.updateCentroids(clusters)
        return centroids

    def euclideanDistance(self, datapoint):
        """
        Finds the euclidean distance between a given point and all datpoints.
        """
        return np.linalg.norm(datapoint - self.data, axis=1)
    
    def assignToClusters(self, centroids):
        """
        Assigns each data point to the closest centroid.
        """
        distances = np.apply_along_axis(self.euclideanDistance, 1, centroids)
        closestCentroidIndices = np.argmin(distances, axis=0)
        clusters = [[] for _ in range(len(centroids))]
        for index, label in enumerate(closestCentroidIndices):
            clusters[label].append(self.data[index])
        return clusters

    def updateCentroids(self, clusters):
        """
        Updates the centroids by taking the mean of all data points in each cluster.
        """
        return np.array([np.mean(cluster, axis=0) for cluster in clusters])