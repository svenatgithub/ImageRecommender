import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean
from PIL import Image
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from config import PATH_TO_SSD, RGB_PATH, HSV_PATH
import plotly.express as px
import umap
from sklearn.cluster import KMeans, AgglomerativeClustering
import plotly.io as pio
from difflib import SequenceMatcher

os.environ['OMP_NUM_THREADS'] = '14'

def load_image_paths(PATH_TO_SSD):
    df = pd.read_csv(PATH_TO_SSD, index_col=0)
    print(f"Image paths loaded. Shape: {df.shape}")
    return df

def compute_histogram(RGB_PATH, color_space='RGB'):
    img = Image.open(RGB_PATH)
    if color_space == 'HSV':
        img = img.convert('HSV')
    histogram = img.histogram()
    return np.array(histogram) / np.sum(histogram)

print("Computing HSV histograms for UMAP...")
hsv_histograms = [compute_histogram(path, 'HSV') for path in PATH_TO_SSD.iloc[:, 0]]
hsv_histograms = np.array(hsv_histograms)

print("Applying UMAP on HSV histograms...")
umap_hsv, umap_model_hsv = apply_umap(hsv_histograms, n_components=3)

n_clusters = 12
print(f"Applying K-Means clustering with {n_clusters} clusters on HSV UMAP...")
kmeans_hsv = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels_hsv = kmeans_hsv.fit_predict(umap_hsv)

print("Plotting K-Means clustering results for HSV...")
plotly_3d_umap(umap_hsv, kmeans_labels_hsv, "Interactive 3D UMAP with K-Means Clustering (HSV)")

print(f"Applying Agglomerative Clustering with {n_clusters} clusters on HSV UMAP...")
agg_clustering_hsv = AgglomerativeClustering(n_clusters=n_clusters)
agg_labels_hsv = agg_clustering_hsv.fit_predict(umap_hsv)

print("Plotting Agglomerative Clustering results for HSV...")
plotly_3d_umap(
    umap_hsv,
    agg_labels_hsv,
    f"Interactive 3D UMAP with Agglomerative Clustering (HSV)"
)

def compute_cosine_similarity(hist1, hist2):
    return 1 - cosine(hist1, hist2)

def compute_euclidean_distance(hist1, hist2):
    return euclidean(hist1, hist2)

def string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def calculate_accuracy(target_path, similar_path, histogram_similarity):
    target_base = os.path.splitext(os.path.basename(target_path))[0].split('_')[0]
    similar_base = os.path.splitext(os.path.basename(similar_path))[0].split('_')[0]
    filename_similarity = string_similarity(target_base, similar_base)
    return (histogram_similarity + filename_similarity) / 2

def find_top_similar_images(histograms, target_histogram, image_paths, metric='cosine', top_n=5):
    if metric == 'cosine':
        similarities = [compute_cosine_similarity(target_histogram, hist) for hist in histograms]
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        scores = [similarities[i] for i in top_indices]
    elif metric == 'euclidean':
        distances = [compute_euclidean_distance(target_histogram, hist) for hist in histograms]
        top_indices = np.argsort(distances)[:top_n]
        scores = [1 / (1 + distances[i]) for i in top_indices]  # Convert distance to similarity
    else:
        raise ValueError("Invalid metric. Use 'cosine' or 'euclidean'.")
    
    top_paths = image_paths.iloc[top_indices, 0].tolist()
    return top_indices, top_paths, scores

def plot_images(image_paths, title, target_image_path=None, accuracies=None):
    n_images = len(image_paths) + (1 if target_image_path else 0)
    fig, axes = plt.subplots(1, n_images, figsize=(20, 5))
    plt.suptitle(title)

    if target_image_path:
        target_img = Image.open(target_image_path)
        axes[0].imshow(target_img)
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        start_idx = 1
    else:
        start_idx = 0

    for i, image_path in enumerate(image_paths):
        if os.path.exists(image_path):
            img = Image.open(image_path)
            axes[i + start_idx].imshow(img)
            title = f"Rank {i + 1}"
            if accuracies:
                title += f"\nAcc: {accuracies[i]:.4f}"
            axes[i + start_idx].set_title(title)
            axes[i + start_idx].axis('off')
        else:
            print(f"Image not found: {image_path}")

    plt.tight_layout()
    plt.show()
    plt.close()

def compare_and_plot_top_images(image_paths, target_image_id):
    target_path = image_paths.loc[target_image_id].iloc[0]
    target_rgb_histogram = compute_histogram(target_path, 'RGB')
    target_hsv_histogram = compute_histogram(target_path, 'HSV')
    
    rgb_histograms = [compute_histogram(path, 'RGB') for path in image_paths.iloc[:, 0]]
    hsv_histograms = [compute_histogram(path, 'HSV') for path in image_paths.iloc[:, 0]]
    
    _, top_cosine_paths_rgb, cosine_scores_rgb = find_top_similar_images(rgb_histograms, target_rgb_histogram, image_paths, 'cosine')
    _, top_euclidean_paths_rgb, euclidean_scores_rgb = find_top_similar_images(rgb_histograms, target_rgb_histogram, image_paths, 'euclidean')
    
    _, top_cosine_paths_hsv, cosine_scores_hsv = find_top_similar_images(hsv_histograms, target_hsv_histogram, image_paths, 'cosine')
    _, top_euclidean_paths_hsv, euclidean_scores_hsv = find_top_similar_images(hsv_histograms, target_hsv_histogram, image_paths, 'euclidean')

    cosine_accuracies_rgb = [calculate_accuracy(target_path, path, score) for path, score in zip(top_cosine_paths_rgb, cosine_scores_rgb)]
    euclidean_accuracies_rgb = [calculate_accuracy(target_path, path, score) for path, score in zip(top_euclidean_paths_rgb, euclidean_scores_rgb)]
    cosine_accuracies_hsv = [calculate_accuracy(target_path, path, score) for path, score in zip(top_cosine_paths_hsv, cosine_scores_hsv)]
    euclidean_accuracies_hsv = [calculate_accuracy(target_path, path, score) for path, score in zip(top_euclidean_paths_hsv, euclidean_scores_hsv)]

    plot_images(top_cosine_paths_rgb, "Top 5 Images by RGB Cosine Similarity", target_path, cosine_accuracies_rgb)
    plot_images(top_euclidean_paths_rgb, "Top 5 Images by RGB Euclidean Distance", target_path, euclidean_accuracies_rgb)
    plot_images(top_cosine_paths_hsv, "Top 5 Images by HSV Cosine Similarity", target_path, cosine_accuracies_hsv)
    plot_images(top_euclidean_paths_hsv, "Top 5 Images by HSV Euclidean Distance", target_path, euclidean_accuracies_hsv)

    return (cosine_accuracies_rgb, euclidean_accuracies_rgb, cosine_accuracies_hsv, euclidean_accuracies_hsv)

def apply_umap(histograms, n_components=3, n_neighbors=12, min_dist=0.1):
    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        n_jobs=-1,
    )
    reduced_histograms = umap_model.fit_transform(histograms)
    return reduced_histograms, umap_model

def plotly_3d_umap(histograms, labels, title):
    fig = px.scatter_3d(
        x=histograms[:, 0],
        y=histograms[:, 1],
        z=histograms[:, 2],
        color=labels,
        title=title,
        labels={"x": "UMAP 1", "y": "UMAP 2", "z": "UMAP 3"},
        color_continuous_scale="Spectral",
    )
    fig.update_traces(marker=dict(size=3))
    pio.show(fig)

if __name__ == "__main__":
    try:
        image_paths = load_image_paths(RGB_PATH)

        target_image_id = image_paths.index[0]
        print(f"Using target image ID: {target_image_id}")

        accuracies = compare_and_plot_top_images(image_paths, target_image_id)

        print("RGB Cosine Accuracies:", accuracies[0])
        print("RGB Euclidean Accuracies:", accuracies[1])
        print("HSV Cosine Accuracies:", accuracies[2])
        print("HSV Euclidean Accuracies:", accuracies[3])

        print("Computing RGB histograms for UMAP...")
        rgb_histograms = [compute_histogram(path, 'RGB') for path in image_paths.iloc[:, 0]]
        rgb_histograms = np.array(rgb_histograms)

        print("Applying UMAP on RGB histograms...")
        umap_rgb, umap_model_rgb = apply_umap(rgb_histograms, n_components=3)

        n_clusters = 12
        print(f"Applying K-Means clustering with {n_clusters} clusters on RGB UMAP...")
        kmeans_rgb = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels_rgb = kmeans_rgb.fit_predict(umap_rgb)

        print("Plotting K-Means clustering results for RGB...")
        plotly_3d_umap(umap_rgb, kmeans_labels_rgb, "Interactive 3D UMAP with K-Means Clustering (RGB)")

        print(f"Applying Agglomerative Clustering with {n_clusters} clusters on RGB UMAP...")
        agg_clustering_rgb = AgglomerativeClustering(n_clusters=n_clusters)
        agg_labels_rgb = agg_clustering_rgb.fit_predict(umap_rgb)

        print("Plotting Agglomerative Clustering results for RGB...")
        plotly_3d_umap(
            umap_rgb,
            agg_labels_rgb,
            f"Interactive 3D UMAP with Agglomerative Clustering (RGB)"
        )

        # Insert HSV Processing Here
        print("Computing HSV histograms for UMAP...")
        hsv_histograms = [compute_histogram(path, 'HSV') for path in image_paths.iloc[:, 0]]
        hsv_histograms = np.array(hsv_histograms)

        print("Applying UMAP on HSV histograms...")
        umap_hsv, umap_model_hsv = apply_umap(hsv_histograms, n_components=3)

        print(f"Applying K-Means clustering with {n_clusters} clusters on HSV UMAP...")
        kmeans_hsv = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels_hsv = kmeans_hsv.fit_predict(umap_hsv)

        print("Plotting K-Means clustering results for HSV...")
        plotly_3d_umap(umap_hsv, kmeans_labels_hsv, "Interactive 3D UMAP with K-Means Clustering (HSV)")

        print(f"Applying Agglomerative Clustering with {n_clusters} clusters on HSV UMAP...")
        agg_clustering_hsv = AgglomerativeClustering(n_clusters=n_clusters)
        agg_labels_hsv = agg_clustering_hsv.fit_predict(umap_hsv)

        print("Plotting Agglomerative Clustering results for HSV...")
        plotly_3d_umap(
            umap_hsv,
            agg_labels_hsv,
            f"Interactive 3D UMAP with Agglomerative Clustering (HSV)"
        )
        
        print("Script completed successfully.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
