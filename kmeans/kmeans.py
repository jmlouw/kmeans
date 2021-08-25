import numpy as np
import scipy

class kmeans:
    """
    A class used to perform the k-means algorithm

    ...

    Attributes
    ----------
    n_clusters: int
        the number of clusters to group the data points

    Methods
    -------
    fit(data)
        Assigns clusters to the data points.
    """
    def __init__(self, n_clusters = 3):
        self.n_clusters = n_clusters
        
    def random_assignment_of_clusters(self, data):
        """
        
        Assigns a random clusters to the data samples.
        Returns a np.array of shape(n,)
        """
        return np.random.randint(low=0, high=self.n_clusters, size=len(data[:, 0]))         
    
    def calculate_centroids(self, data, clusters):
        """
        Calculates the centroids of each cluster
        Parameters:
        data: np.array:
            The dataset with a shape=(n, 2)
        clusters: np.array
            The clusters (shape=(n,)) that are assigned to the data points.
        
        Returns
        -------
        List[np.array]
            A list with a centroid for each cluster
        
        """
        centroids = []
        for i in range(self.n_clusters):
            mask = clusters == i       
            centroids.append(np.mean(data[mask, :], axis = 0))       
        return centroids
    
    def assign_clusters(self, data, centroids):
        """
        Assigns new clusters to the data points:
        Parameters
        ----------
        data: np.array
            The dataset with a shape=(n, 2)
        
        Returns
        -------
        centroids: np.array
            A vector with the cluster for each data sample (shape=(n,))      
        
        """
        dist = scipy.spatial.distance.cdist(centroids,data) 
        return np.argmin(dist, axis = 0)
        
    def fit(self, data):
        """Assigns clusters to a data set

        Parameters
        ----------
        data: np.array
            The dataset with a shape=(n, 2)

        Returns
        -------
        new_clusters: np.array
            a numpy array with the a cluster for each data point
        centroids: List[np.array]
            a list with a centroid for each cluster
        """        
        """
        Takes a 2D numpy array of shape=(n, 2), where n is the number of samples in the dataset.
        
        Returns a numpy array that has shape=(n,) that contains the cluster of each data sample,
        where the third channel is used to indicate the cluster of each point in the data set.
        """
        old_clusters = self.random_assignment_of_clusters(data)
        while True:
            centroids = self.calculate_centroids(data, old_clusters)
            new_clusters = self.assign_clusters(data, centroids)
            if (new_clusters==old_clusters).all():
                break
            old_clusters=new_clusters        
        return new_clusters, centroids