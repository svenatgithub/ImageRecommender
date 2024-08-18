import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from config import PICKLE_PATH, RGB_PATH, HSV_PATH, PATH_TO_SSD
import os

def load_embeddings_from_pickle(PICKLE_PATH):
    """Load embeddings from a pickle file."""
    with open(PICKLE_PATH, 'rb') as f:
        data = pickle.load(f)

    # Assuming the embeddings are stored under a specific key, e.g., 'embeddings'
    if isinstance(data, dict) and 'embeddings' in data:
        embeddings = data['embeddings']
    else:
        raise ValueError("Pickle file does not contain the expected embeddings dictionary.")
    
    return embeddings

def load_histograms_from_csv(RGB_PATH, HSV_PATH):
    """Load RGB and HSV histograms from CSV files."""
    rgb_histograms = pd.read_csv(RGB_PATH)
    hsv_histograms = pd.read_csv(HSV_PATH)
    return rgb_histograms.values, hsv_histograms.values

def find_top_5_similar_images(embeddings, rgb_histograms, hsv_histograms, image_index):
    """Find the top 5 most similar images based on embeddings, RGB histograms, and HSV histograms."""
    if isinstance(embeddings, dict):
        embeddings = np.array(list(embeddings.values()))  # Convert dict values to numpy array

    knn_embeddings = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(embeddings)
    knn_rgb = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(rgb_histograms)
    knn_hsv = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(hsv_histograms)
    
    distances_embeddings, indices_embeddings = knn_embeddings.kneighbors([embeddings[image_index]])
    distances_rgb, indices_rgb = knn_rgb.kneighbors([rgb_histograms[image_index]])
    distances_hsv, indices_hsv = knn_hsv.kneighbors([hsv_histograms[image_index]])

    # Combine indices from all methods to determine top 5 unique images
    combined_indices = np.concatenate((indices_embeddings[0], indices_rgb[0], indices_hsv[0]))
    unique_indices, counts = np.unique(combined_indices, return_counts=True)
    top_5_indices = unique_indices[np.argsort(-counts)][:5]

    return top_5_indices

def plot_top_5_images(top_5_indices, PATH_TO_SSD, image_paths):
    """Plot the top 5 most similar images."""
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(top_5_indices):
        img = Image.open(os.path.join(PATH_TO_SSD, image_paths[idx]))
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.title(f"Rank {i+1}")
        plt.axis('off')
    plt.show()

def main():
    # Load embeddings from pickle file
    pickle_path = os.path.join(PICKLE_PATH, 'embeddings.pkl')
    embeddings = load_embeddings_from_pickle(pickle_path)

    # Load RGB and HSV histograms from CSV files
    rgb_histograms, hsv_histograms = load_histograms_from_csv(RGB_PATH, HSV_PATH)

    # List of image paths
    image_paths = [f for f in os.listdir(PATH_TO_SSD) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Choose an image index to compare against (for testing, you can pick any index)
    image_index = 0  # Change this index as needed

    # Find top 5 most similar images
    top_5_indices = find_top_5_similar_images(embeddings, rgb_histograms, hsv_histograms, image_index)

    # Plot the top 5 most similar images
    plot_top_5_images(top_5_indices, PATH_TO_SSD, image_paths)

    print("KNN creation, similarity ranking, and plotting complete.")

if __name__ == "__main__":
    main()
