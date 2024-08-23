import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean
from PIL import Image
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from config import PATH_TO_SSD, FINAL_EMBEDDINGS_PATH

import plotly.express as px
import umap
from sklearn.cluster import KMeans, AgglomerativeClustering
import plotly.io as pio

from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from difflib import SequenceMatcher

os.environ['OMP_NUM_THREADS'] = '14'

def load_data(FINAL_EMBEDDINGS_PATH):
    print(f"Loading data from {FINAL_EMBEDDINGS_PATH}")
    with open(FINAL_EMBEDDINGS_PATH, 'rb') as f:
        data = pickle.load(f)
    print(f"Data loaded. Number of images: {len(data)}")
    print(f"Sample key: {list(data.keys())[0]}")
    print(f"Sample value shape: {data[list(data.keys())[0]].shape}")
    return data

def string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def calculate_accuracy(embeddings, target_image_id, similar_images, metric='cosine'):
    target_embedding = embeddings[target_image_id]
    target_base = os.path.splitext(target_image_id)[0].split('_')[0]
    
    accuracies = []
    for similar_image_id in similar_images:
        if metric == 'cosine':
            distance = cosine(target_embedding, embeddings[similar_image_id])
            similarity = 1 - distance
        elif metric == 'euclidean':
            distance = euclidean(target_embedding, embeddings[similar_image_id])
            similarity = 1 / (1 + distance)
        else:
            raise ValueError("Invalid metric. Use 'cosine' or 'euclidean'.")
        
        similar_base = os.path.splitext(similar_image_id)[0].split('_')[0]
        filename_similarity = string_similarity(target_base, similar_base)
        
        combined_accuracy = (similarity + filename_similarity) / 2
        accuracies.append((similar_image_id, combined_accuracy))
    
    return accuracies

def find_top_similar_images(embedding_values, target_embedding, image_ids, metric='cosine', top_n=5):
    if metric == 'cosine':
        similarities = cosine_similarity([target_embedding], embedding_values)[0]
        top_indices = similarities.argsort()[-top_n:][::-1]
    elif metric == 'euclidean':
        distances = np.array([euclidean(target_embedding, emb) for emb in embedding_values])
        top_indices = distances.argsort()[:top_n]
    else:
        raise ValueError("Invalid metric. Use 'cosine' or 'euclidean'.")
    
    top_image_ids = [image_ids[i] for i in top_indices]
    return top_indices, top_image_ids

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

def compare_and_plot_top_images(embeddings, target_image_id):
    target_embedding = embeddings[target_image_id]
    embedding_values = np.array(list(embeddings.values()))
    image_ids = list(embeddings.keys())

    _, top_cosine_ids = find_top_similar_images(embedding_values, target_embedding, image_ids, 'cosine', top_n=5)
    _, top_euclidean_ids = find_top_similar_images(embedding_values, target_embedding, image_ids, 'euclidean', top_n=5)

    plot_images(top_cosine_ids, "Input Image and Top 5 Images by Cosine Similarity", target_image_id)
    plot_images(top_euclidean_ids, "Input Image and Top 5 Images by Euclidean Distance", target_image_id)
    
    cosine_accuracies = calculate_accuracy(embeddings, target_image_id, top_cosine_ids, 'cosine')
    euclidean_accuracies = calculate_accuracy(embeddings, target_image_id, top_euclidean_ids, 'euclidean')
    
    return cosine_accuracies, euclidean_accuracies

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

def apply_kernel_pca(embeddings, n_components=100):
    print("Applying Kernel PCA...")
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    
    kpca = KernelPCA(n_components=n_components, kernel='rbf', n_jobs=-1, random_state=42)
    reduced_embeddings = kpca.fit_transform(scaled_embeddings)
    
    print(f"Embeddings reduced from {embeddings.shape[1]} to {n_components} dimensions")
    return reduced_embeddings, kpca

if __name__ == "__main__":
    embeddings = load_data(FINAL_EMBEDDINGS_PATH)
    if not embeddings:
        print("Failed to load data. Please check the pickle file and its structure.")
        exit(1)

    uuids = list(embeddings.keys())
    embedding_values = np.array(list(embeddings.values()))

    kpca_embeddings, kpca_model = apply_kernel_pca(embedding_values, n_components=100)

    target_image_id = uuids[10]
    print(f"Using target image ID: {target_image_id}")

    reduced_embeddings = {uuid: emb for uuid, emb in zip(uuids, kpca_embeddings)}

    cosine_accuracies, euclidean_accuracies = compare_and_plot_top_images(reduced_embeddings, target_image_id)

    print("Cosine similarity accuracies:")
    for image_id, accuracy in cosine_accuracies:
        print(f"{image_id}: {accuracy:.4f}")

    print("\nEuclidean distance accuracies:")
    for image_id, accuracy in euclidean_accuracies:
        print(f"{image_id}: {accuracy:.4f}")

    print("Applying UMAP...")
    umap_embeddings, umap_model = apply_umap(kpca_embeddings, n_components=3, n_neighbors=12, min_dist=0.1)

    n_clusters = 12
    print(f"Applying K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(kpca_embeddings)

    print("Plotting K-Means clustering results...")
    plotly_3d_umap(umap_embeddings, kmeans_labels, "Interactive 3D UMAP with K-Means Clustering")

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
