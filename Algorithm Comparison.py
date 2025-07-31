## Importing Datasets and libraries ##
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def plot_clusters(X, labels, title):
    plt.figure(figsize=(5, 4))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="Set1", s=30, edgecolor="k")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

## Dataset 1: make_moons ##
X1, _ = make_moons(n_samples=300, noise=0.05)

## Dataset 2: make_blobs with varying density ##
X2, _ = make_blobs(n_samples=300, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=42)

## Applying clustering on Dataset 1 ##
models = {
    "DBSCAN": DBSCAN(eps=0.3, min_samples=5),
    "KMeans": KMeans(n_clusters=2, random_state=42),
    "Agglomerative": AgglomerativeClustering(n_clusters=2)
}

for name, model in models.items():
    labels = model.fit_predict(X1)
    plot_clusters(X1, labels, f"{name} on make_moons")

## Applying clustering on Dataset 2 ##
models2 = {
    "DBSCAN": DBSCAN(eps=0.9, min_samples=5),
    "KMeans": KMeans(n_clusters=3, random_state=42),
    "Agglomerative": AgglomerativeClustering(n_clusters=3)
}

for name, model in models2.items():
    labels = model.fit_predict(X2)
    plot_clusters(X2, labels, f"{name} on make_blobs")
