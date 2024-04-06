from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np

def find_optimal_clusters_silhouette(data, max_k):
    silhouette_scores = []
    K = range(2, max_k+1)  # At least 2 clusters are needed to compute silhouette score

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        labels = kmeans.labels_
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
        print(f"Silhouette score for {k} clusters: {score}")

    optimal_k = np.argmax(silhouette_scores) + 2  # +2 because range starts at 2
    print(f"Optimal number of clusters: {optimal_k}")

    return optimal_k