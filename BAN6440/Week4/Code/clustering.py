"""
Phrase Clustering Analysis
This script performs K-means clustering on text phrase data to identify distinct patterns.
It includes data downloading, preprocessing, feature extraction, clustering, and visualization.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import subprocess
import logging
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_with_cli(bucket_name: str, file_name: str, output_file: str) -> None:
    """
    Downloads a file from an S3 bucket using AWS CLI with no-sign-request.
    """
    try:
        command = (
            f"aws s3 cp s3://{bucket_name}/{file_name} {output_file} --no-sign-request"
        )
        subprocess.run(command, shell=True, check=True)
        logger.info(f"File {file_name} downloaded successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading the file: {e}")
        raise


def load_json_data(file_path: str) -> Dict:
    """
    Load and validate JSON data from file.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        logger.info(f"Successfully loaded data from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error("Invalid JSON format")
        raise


def extract_features(phrases: List[List[str]]) -> pd.DataFrame:
    """
    Extract numerical features from text phrases.
    """
    clean_phrases = [" ".join(phrase).lower() for phrase in phrases]
    word_counts = [len(phrase.split()) for phrase in clean_phrases]
    char_counts = [len(phrase) for phrase in clean_phrases]
    return pd.DataFrame({"word_count": word_counts, "char_count": char_counts})


def process_data(data: Dict) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Process raw data and prepare for clustering.
    """
    all_features = []
    categories = []
    groups = []

    for category in data:
        for group_name, phrases in data[category].items():
            features = extract_features(phrases)
            all_features.append(features)
            categories.extend([category] * len(phrases))
            groups.extend([group_name] * len(phrases))

    all_features_df = pd.concat(all_features, ignore_index=True)
    all_features_df["category"] = categories
    all_features_df["group"] = groups

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(
        all_features_df[["word_count", "char_count"]]
    )

    return all_features_df, scaled_features


def plot_elbow_curve(scaled_features: np.ndarray) -> None:
    """
    Plot elbow curve to determine optimal number of clusters.
    """
    inertias = []
    K = range(1, 11)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(K, inertias, "bx-")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method For Optimal k")
    plt.savefig("elbow_curve.png")  # Save the plot
    plt.show()


def perform_clustering(scaled_features: np.ndarray, n_clusters: int = 4) -> np.ndarray:
    """
    Perform K-means clustering.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    logger.info(f"Successfully performed clustering with {n_clusters} clusters")
    return clusters


def create_visualizations(
    scaled_features: np.ndarray, clusters: np.ndarray, all_features_df: pd.DataFrame
) -> None:
    """
    Create various visualizations of the clustering results.
    """
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        scaled_features[:, 0], scaled_features[:, 1], c=clusters, cmap="viridis"
    )
    plt.xlabel("Word Count (scaled)")
    plt.ylabel("Character Count (scaled)")
    plt.title("Phrase Clusters")
    plt.colorbar(scatter)
    plt.savefig("scatter_clusters.png")  # Save the plot
    plt.show()

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        pca_features[:, 0], pca_features[:, 1], c=clusters, cmap="viridis"
    )
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.title("Phrase Clusters (PCA)")
    plt.colorbar(scatter)
    plt.savefig("pca_clusters.png")  # Save the plot
    plt.show()


def main():
    """
    Main function to orchestrate the clustering analysis.
    """
    try:
        # Step 1: Download the file
        bucket_name = "amazon-phrase-clustering"
        file_name = "phrase-clustering-dataset.json"
        output_file = "phrase-clustering-dataset.json"
        download_with_cli(bucket_name, file_name, output_file)

        # Step 2: Load the data
        data = load_json_data(output_file)

        # Step 3: Process the data
        all_features_df, scaled_features = process_data(data)

        # Step 4: Plot the elbow curve
        plot_elbow_curve(scaled_features)

        # Step 5: Perform kmeans clustering
        clusters = perform_clustering(scaled_features)

        # Step 6: Add cluster assignments to the pandas DataFrame
        all_features_df["Cluster"] = clusters

        # Step 7: Create the visualizations
        create_visualizations(scaled_features, clusters, all_features_df)

        # Step 8: Print the cluster statistics
        print("\nCluster Statistics:")
        cluster_stats = (
            all_features_df.groupby("Cluster")
            .agg({"word_count": ["mean", "count"], "char_count": "mean"})
            .round(2)
        )
        print(cluster_stats)

        logger.info("Analysis completed successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
