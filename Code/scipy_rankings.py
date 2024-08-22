import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean
from PIL import Image
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from config import PATH_TO_SSD, RGB_PATH, HSV_PATH, FINAL_EMBEDDINGS_PATH

import plotly.express as px
import umap
from sklearn.cluster import KMeans, AgglomerativeClustering
from joblib import Parallel, delayed
import sqlite3
import plotly.io as pio

# New imports for Kernel PCA
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

def load_embeddings(path):
    print(f"Loading embeddings from {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Embeddings loaded. Keys in the loaded data: {data.keys()}")
    return data

def load_histograms(path):
    print(f"Loading histograms from {path}")
    df = pd.read_csv(path, header=0)
    print(f"Histograms loaded. Shape: {df.shape}")
    return df

def compute_cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

def compute_euclidean_distance(embedding1, embedding2):
    return euclidean(embedding1, embedding2)

def find_top_similar_images(embedding_values, target_embedding, metric='cosine', top_n=5):
    if metric == 'cosine':
        similarities = cosine_similarity([target_embedding], embedding_values)[0]
        top_indices = similarities.argsort()[-top_n:][::-1]
    elif metric == 'euclidean':
        distances = np.array([euclidean(target_embedding, emb) for emb in embedding_values])
        top_indices = distances.argsort()[:top_n]
    else:
        raise ValueError("Invalid metric. Use 'cosine' or 'euclidean'.")
    return top_indices

def plot_images(image_paths, title, target_image_path=None):
    n_images = len(image_paths) + (1 if target_image_path else 0)
    fig, axes = plt.subplots(1, n_images, figsize=(20, 5))
    plt.suptitle(title)

    if target_image_path:
        target_img = Image.open(os.path.join(PATH_TO_SSD, target_image_path))
        axes[0].imshow(target_img)
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        start_idx = 1
    else:
        start_idx = 0

    for i, image_path in enumerate(image_paths):
        full_path = os.path.join(PATH_TO_SSD, image_path)
        if os.path.exists(full_path):
            img = Image.open(full_path)
            axes[i + start_idx].imshow(img)
            axes[i + start_idx].set_title(f"Rank {i + 1}")
            axes[i + start_idx].axis('off')
        else:
            print(f"Image not found: {full_path}")

    plt.tight_layout()
    plt.show()
    plt.close()

def compare_and_plot_top_images(embeddings, rgb_histograms, hsv_histograms, target_image_id):
    target_embedding = embeddings[target_image_id]
    embedding_values = np.array(list(embeddings.values()))
    image_ids = list(embeddings.keys())

    top_cosine_indices = find_top_similar_images(embedding_values, target_embedding, 'cosine')
    top_euclidean_indices = find_top_similar_images(embedding_values, target_embedding, 'euclidean')

    top_cosine_paths = [image_ids[i] for i in top_cosine_indices]
    top_euclidean_paths = [image_ids[i] for i in top_euclidean_indices]

    plot_images(top_cosine_paths, "Input Image and Top 5 Images by Cosine Similarity", target_image_id)
    plot_images(top_euclidean_paths, "Input Image and Top 5 Images by Euclidean Distance", target_image_id)

def apply_umap(embeddings, n_components=3, n_neighbors=12, min_dist=0.1):
    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        n_jobs=-1,
    )
    reduced_embeddings = umap_model.fit_transform(embeddings)
    return reduced_embeddings, umap_model

def plotly_3d_umap(embeddings, labels, title):
    fig = px.scatter_3d(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        z=embeddings[:, 2],
        color=labels,
        title=title,
        labels={"x": "UMAP 1", "y": "UMAP 2", "z": "UMAP 3"},
        color_continuous_scale="Spectral",
    )
    fig.update_traces(marker=dict(size=3))
    pio.show(fig)

# New function for Kernel PCA
def apply_kernel_pca(embeddings, n_components=100):
    print("Applying Kernel PCA...")
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    
    kpca = KernelPCA(n_components=n_components, kernel='rbf', n_jobs=-1, random_state=42)
    reduced_embeddings = kpca.fit_transform(scaled_embeddings)
    
    print(f"Embeddings reduced from {embeddings.shape[1]} to {n_components} dimensions")
    return reduced_embeddings, kpca

if __name__ == "__main__":
    embeddings = load_embeddings(FINAL_EMBEDDINGS_PATH)
    if not embeddings:
        print("Failed to load embeddings. Please check the pickle file and its structure.")
        exit(1)

    rgb_histograms = load_histograms(RGB_PATH)
    hsv_histograms = load_histograms(HSV_PATH)

    # Convert embeddings to numpy array
    uuids = list(embeddings.keys())
    embedding_values = np.array(list(embeddings.values()))

    # Apply Kernel PCA
    kpca_embeddings, kpca_model = apply_kernel_pca(embedding_values, n_components=100)

    # Use the first image as an example
    target_image_id = uuids[0]
    print(f"Using target image ID: {target_image_id}")

    # Update embeddings dictionary with reduced embeddings
    reduced_embeddings = {uuid: emb for uuid, emb in zip(uuids, kpca_embeddings)}

    compare_and_plot_top_images(reduced_embeddings, rgb_histograms, hsv_histograms, target_image_id)

    print("Applying UMAP...")
    umap_embeddings, umap_model = apply_umap(kpca_embeddings, n_components=3, n_neighbors=12, min_dist=0.1)

    # K-Means clustering
    n_clusters = 12
    print(f"Applying K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(kpca_embeddings)

    print("Plotting K-Means clustering results...")
    plotly_3d_umap(umap_embeddings, kmeans_labels, "Interactive 3D UMAP with K-Means Clustering")

    # Agglomerative Clustering
    print(f"Applying Agglomerative Clustering with {n_clusters} clusters...")
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels = agg_clustering.fit_predict(kpca_embeddings)

    print("Plotting Agglomerative Clustering results...")
    plotly_3d_umap(
        umap_embeddings,
        agg_labels,
        f"Interactive 3D UMAP with Agglomerative Clustering (n_clusters={n_clusters})",
    )

    print("Script completed successfully.")
