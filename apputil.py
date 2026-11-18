import numpy as np
import seaborn as sns
from time import time
from sklearn.cluster import KMeans


def kmeans(X, k):
    """Run k-means on array X and return (centroids, labels)"""
    # create and fit the k-means model
    model = KMeans(n_clusters=k)
    model.fit(X)

    centroids = model.cluster_centers_
    labels = model.labels_

    return centroids, labels


# load the diamonds dataset and keep only the numeric columns
diamonds = sns.load_dataset("diamonds")
diamonds_numeric = diamonds.select_dtypes(include="number")


def kmeans_diamonds(n, k):
    """Run kmeans on the first n rows of the diamonds data"""
    # take the first n rows as a numpy array
    X = diamonds_numeric.head(n).to_numpy()
    return kmeans(X, k)


def kmeans_timer(n, k, n_iter=5):
    """Run kmeans_diamonds(n, k) n times and return the average runtime"""
    times = []
    for _ in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        times.append(time() - start)
    # average runtime across all runs
    return np.mean(times)
