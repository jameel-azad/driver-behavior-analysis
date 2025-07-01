import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def run_pca_kmeans(data, n_clusters=8):
    # no. of clusters decided based on analysis over different number of clusters using methods like Silhouette Score, WSS and BSS/TSS

    # retains 99% variance
    pca = PCA(n_components=30, random_state=0)
    reduced_data = pca.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    kmeans.fit(reduced_data)

    cluster_centers_reduced = kmeans.cluster_centers_
    approx_centers = pca.inverse_transform(cluster_centers_reduced)

    return kmeans, approx_centers, pca
