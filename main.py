import pandas as pd
from scripts.clustering import run_pca_kmeans
from scripts.inverse_transform import invert_cluster_centers
from scripts.plotting import plot_cluster_profiles

def main():
    data = pd.read_csv('processed_data/processed_data.csv')
    kmeans, approx_centers, pca = run_pca_kmeans(data)
    original_centers = invert_cluster_centers(approx_centers)
    plot_cluster_profiles(original_centers)

if __name__ == '__main__':
    main()
