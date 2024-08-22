import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from config import FINAL_EMBEDDINGS_PATH, RGB_PATH, HSV_PATH, PATH_TO_SSD
import os
from PIL import Image
import csv

def load_embeddings_from_pickle(FINAL_EMBEDDINGS_PATH):
    print(f"Loading embeddings from {FINAL_EMBEDDINGS_PATH}")
    try:
        with open(FINAL_EMBEDDINGS_PATH, 'rb') as f:
            data = pickle.load(f)
        print(f"Pickle file loaded successfully")
        print(f"Type of loaded data: {type(data)}")
        
        if isinstance(data, dict):
            print("Loaded data is a dictionary")
            if 'embeddings_data' in data:
                embeddings_data = data['embeddings_data']
            else:
                embeddings_data = data  # Assume the whole dict is the embeddings data
            
            print(f"Type of embeddings_data: {type(embeddings_data)}")
            if isinstance(embeddings_data, dict):
                print(f"Number of embeddings: {len(embeddings_data)}")
                embeddings = list(embeddings_data.values())
                if embeddings:
                    return np.array(embeddings)
                else:
                    print("No valid embeddings found in the data")
            elif isinstance(embeddings_data, list):
                print(f"Number of embeddings: {len(embeddings_data)}")
                if len(embeddings_data) > 0:
                    embeddings = [item['embedding'] for item in embeddings_data if 'embedding' in item]
                    if embeddings:
                        return np.array(embeddings)
                    else:
                        print("No valid embeddings found in the data")
                else:
                    print("Embeddings data list is empty")
            else:
                print("Embeddings data is not a list or dictionary")
        elif isinstance(data, list):
            print(f"Number of embeddings: {len(data)}")
            if len(data) > 0 and isinstance(data[0], dict):
                embeddings = [item['embedding'] for item in data if 'embedding' in item]
                if embeddings:
                    return np.array(embeddings)
                else:
                    print("No valid embeddings found in the data")
            else:
                print("Data format is not as expected")
        else:
            print("Loaded data is neither a list nor a dictionary")
        
        return None
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None

def load_histograms_from_csv(RGB_PATH, HSV_PATH):
    print(f"Loading RGB histograms from {RGB_PATH}")
    rgb_histograms = pd.read_csv(RGB_PATH, header=0)
    print(f"RGB histograms shape: {rgb_histograms.shape}")
    print(f"RGB histograms head:\n{rgb_histograms.head()}")
    print(f"RGB histograms info:\n{rgb_histograms.info()}")
    
    print(f"\nLoading HSV histograms from {HSV_PATH}")
    hsv_histograms = pd.read_csv(HSV_PATH, header=0)
    print(f"HSV histograms shape: {hsv_histograms.shape}")
    print(f"HSV histograms head:\n{hsv_histograms.head()}")
    print(f"HSV histograms info:\n{hsv_histograms.info()}")
    
    rgb_histograms = clean_histogram_data(rgb_histograms)
    hsv_histograms = clean_histogram_data(hsv_histograms)
    
    if rgb_histograms.empty or hsv_histograms.empty:
        raise ValueError("One or both histogram DataFrames are empty after cleaning")
    
    # Return only the histogram data (excluding metadata)
    return rgb_histograms.iloc[:, 2:].values, hsv_histograms.iloc[:, 2:].values

def clean_histogram_data(df):
    print("Unique values before cleaning:")
    print(df.nunique().describe())
    
    # Separate metadata and histogram data
    metadata = df.iloc[:, :2]
    histogram_data = df.iloc[:, 2:]
    
    # Convert histogram data to numeric, replacing errors with NaN
    histogram_data = histogram_data.apply(pd.to_numeric, errors='coerce')
    print(f"Shape after conversion: {histogram_data.shape}")
    print(f"Number of NaN values: {histogram_data.isna().sum().sum()}")
    
    # Drop rows with any NaN values in the histogram data
    valid_rows = histogram_data.dropna().index
    
    # Use the valid rows to filter both metadata and histogram data
    clean_metadata = metadata.loc[valid_rows]
    clean_histogram_data = histogram_data.loc[valid_rows]
    
    # Combine metadata and clean histogram data
    clean_df = pd.concat([clean_metadata, clean_histogram_data], axis=1)
    print(f"Shape after cleaning: {clean_df.shape}")
    print(f"Info after cleaning:\n{clean_df.info()}")
    
    if clean_df.empty:
        print("WARNING: DataFrame is empty after cleaning!")
    return clean_df

def find_top_5_similar_images(embeddings, rgb_histograms, hsv_histograms, image_index):
    if isinstance(embeddings, dict):
        embeddings = np.array(list(embeddings.values()))  # Convert dict values to numpy array

    if embeddings.size == 0 or rgb_histograms.size == 0 or hsv_histograms.size == 0:
        raise ValueError("One or more input arrays are empty")

    if image_index >= len(embeddings):
        raise ValueError(f"Image index {image_index} is out of bounds for embeddings array of length {len(embeddings)}")

    knn_embeddings = NearestNeighbors(n_neighbors=min(5, len(embeddings)), algorithm='ball_tree').fit(embeddings)
    knn_rgb = NearestNeighbors(n_neighbors=min(5, len(rgb_histograms)), algorithm='ball_tree').fit(rgb_histograms)
    knn_hsv = NearestNeighbors(n_neighbors=min(5, len(hsv_histograms)), algorithm='ball_tree').fit(hsv_histograms)
    
    distances_embeddings, indices_embeddings = knn_embeddings.kneighbors([embeddings[image_index]])
    distances_rgb, indices_rgb = knn_rgb.kneighbors([rgb_histograms[image_index]])
    distances_hsv, indices_hsv = knn_hsv.kneighbors([hsv_histograms[image_index]])

    combined_indices = np.concatenate((indices_embeddings[0], indices_rgb[0], indices_hsv[0]))
    unique_indices, counts = np.unique(combined_indices, return_counts=True)
    top_5_indices = unique_indices[np.argsort(-counts)][:5]
    return top_5_indices

def plot_top_5_images(top_5_indices, PATH_TO_SSD, image_paths):
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(top_5_indices):
        img = Image.open(os.path.join(PATH_TO_SSD, image_paths[idx]))
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.title(f"Rank {i+1}")
        plt.axis('off')
    plt.show()

def check_csv_file(file_path):
    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f)
        first_row = next(csv_reader, None)
        if first_row is None:
            print(f"The file {file_path} is empty.")
        else:
            print(f"The file {file_path} contains data. First row: {first_row[:5]}...")

def main():
    try:
        # Check CSV files
        check_csv_file(RGB_PATH)
        check_csv_file(HSV_PATH)

        # Load embeddings from pickle file
        embeddings = load_embeddings_from_pickle(FINAL_EMBEDDINGS_PATH)
        if embeddings is None:
            print("Failed to load embeddings. Please check the pickle file and its path.")
            return

        # Load RGB and HSV histograms from CSV files
        rgb_histograms, hsv_histograms = load_histograms_from_csv(RGB_PATH, HSV_PATH)

        # List of image paths
        image_paths = [f for f in os.listdir(PATH_TO_SSD) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_paths:
            print("No image files found in the specified directory.")
            return

        # Choose an image index to compare against (for testing, you can pick any index)
        image_index = 1  # Change this index as needed

        # Find top 5 most similar images
        top_5_indices = find_top_5_similar_images(embeddings, rgb_histograms, hsv_histograms, image_index)

        # Plot the top 5 most similar images
        plot_top_5_images(top_5_indices, PATH_TO_SSD, image_paths)

        print("KNN creation, similarity ranking, and plotting complete.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 
